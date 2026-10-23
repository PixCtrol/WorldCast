"""
WorldCast anchor-conditioning modules and model_fn.

Trainable external modules injected around the frozen Wan2.1-I2V-14B DiT:

  - CameraProjection: 6-ch Pluecker camera embedding -> DiT token space
    (target trajectory conditioning; also reused for anchor rays via
    --anchor_plucker_use_camera_proj).
  - AnchorChannelAdapter: 1x1 conv packing each variable-resolution anchor
    view [lat16 | ones4 | lat16 | person_mask1] (37ch, native Wan I2V layout
    + mask) back to dit.in_dim (36ch) so the frozen patch_embedding applies.
  - SourceSelfAttentionBank: external self-attention bank attending over
    [anchor tokens | video tokens], per-layer zero-init output projections,
    rotary embeddings reusing the DiT's RoPE (use_dit_rope).
  - WanI2VChannelAdapter: 1x1 conv mapping [native_i2v(36) | render_latent(16)
    | render_mask(1)] (53ch) back to dit.in_dim before patchify.

model_fn_worldcast implements the released inference path: token-concat
anchor prefix + render channel-adapter conditioning + per-view Pluecker
injection + bank residuals on the video tokens.
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from ..models.controlnet import SingleAttentionBlock
from .worldcast_pipeline import (
    WorldCastVideoPipeline,
    _build_anchor_rope,
    WanRotaryPosEmbed,
    sinusoidal_embedding_1d,
)


class CameraProjection(torch.nn.Module):
    """Project 6-ch camera embedding to DiT hidden dim.

    Two-stage projection for memory efficiency:
    1. Conv3d(6 -> mid_dim, kernel=1): lightweight channel projection
    2. Conv3d(mid_dim -> out_dim, kernel=[1,2,2], stride=[1,2,2]): spatial downsampling + expansion

    Input is at latent resolution (H//8, W//8) after temporal+spatial downsampling.
    Output is at post-patchification resolution (H//16, W//16) with out_dim channels,
    matching dit.patch_embedding output spatial dims.
    """

    def __init__(self, in_channels=6, mid_dim=256, out_dim=5120, out_init_std=0.0):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, mid_dim, kernel_size=1, bias=True),
            torch.nn.SiLU(),
            torch.nn.Conv3d(mid_dim, out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=True),
        )
        if out_init_std > 0:
            # Non-zero output init (e.g. std=0.02): the projection participates
            # from step 0. Needed when the consumer's own output layer is
            # zero-init (e.g. a zero-init residual bank blocks gradient flow to
            # its KV tokens until that layer moves off zero).
            torch.nn.init.normal_(self.proj[-1].weight, std=out_init_std)
            torch.nn.init.zeros_(self.proj[-1].bias)
        else:
            torch.nn.init.zeros_(self.proj[-1].weight)
            torch.nn.init.zeros_(self.proj[-1].bias)

    def forward(self, camera_embedding):
        """
        :param camera_embedding: [b, 6, f, h, w] camera features at latent spatial resolution
        :returns: [b, f * h//2 * w//2, out_dim] token-space camera conditioning
        """
        x = self.proj(camera_embedding)              # [b, out_dim, f, h//2, w//2]
        x = x.flatten(2).transpose(1, 2)             # [b, seq, out_dim]
        return x


class AnchorChannelAdapter(nn.Module):
    """1x1x1 conv projecting the packed VAE anchor latent to the DiT input
    channel dim (in_dim, 36 for Wan2.1 I2V 14B), so the anchor reuses the
    (frozen) ``dit.patch_embedding`` cleanly.

    The input is the native Wan I2V reference layout plus one mask channel:
    ``[lat(16) | ones(4) | lat(16) | person_mask(1)]`` = 2*z_dim+5 = 37ch.
    ``init="identity"`` starts the conv as an EXACT pass-through of the first
    out_dim channels with zero weight on the mask channel and zero bias —
    the run starts bit-equal to the plain i2v_style packing, and training
    learns ONLY the mask increment. Requires in_channels == out_dim + 1.

    ``kernel_size=1`` keeps the latent spatial resolution intact; the (1,2,2)
    patchify is still performed exactly once by ``dit.patch_embedding``
    downstream (no double downsample).
    """

    def __init__(self, in_channels=37, out_dim=36, init="identity"):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.proj = nn.Conv3d(in_channels, out_dim, kernel_size=1, bias=True)
        if init == "identity":
            if in_channels != out_dim + 1:
                raise ValueError(
                    f"AnchorChannelAdapter init='identity' needs in_channels == "
                    f"out_dim + 1, got {in_channels} != {out_dim} + 1")
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            with torch.no_grad():
                self.proj.weight.view(out_dim, in_channels)[
                    torch.arange(out_dim), torch.arange(out_dim)] = 1.0

    def forward(self, x):
        # x: [B, in_channels, n_anc, h, w] -> [B, out_dim, n_anc, h, w]
        return self.proj(x)


class SourceSelfAttentionBank(nn.Module):
    """Trainable self-attention blocks injected at intervals inside frozen DiT.

    If inner_dim < dim, tokens are projected dim->inner_dim before self-attn and
    the delta is projected inner_dim->dim back (proj_out zero-init -> starts no-op).
    This drastically reduces params (e.g. dim=5120->inner_dim=1024: ~11x fewer).

    use_dit_rope: reuse the model_fn's FULL-sequence rotary (the same RoPE the
    frozen DiT sees — anchors at their rope-offset temporal positions, e.g.
    25/30/35/40) instead of a bank-own rope. Requires bank head_dim ==
    the DiT rope head_dim (128 for Wan 14B; e.g. inner_dim=1024 -> heads=8), so
    the rope tensor can be shared as-is and the bank's routing learns the same
    azimuth-region addressing the token-concat line trains into DiT self-attn
    itself.
    """

    def __init__(self, num_layers=30, interval=2, dim=3072, ffn_dim=None,
                 num_heads=24, time_embed_dim=3072, inner_dim=None, start_block=0,
                 proj_out_init_std=0.0, use_dit_rope=False, num_layers_is_span=True,
                 per_layer_proj_out=False):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim or dim
        self.use_proj = self.inner_dim < dim
        # num_layers semantics (backward-compat knob):
        #   num_layers_is_span=True  (legacy default): num_layers = DiT block SPAN.
        #       bank layer count = ceil(num_layers / interval); e.g. 20/2 -> 10 layers
        #       injecting at blocks 0,2,...,18.
        #   num_layers_is_span=False: num_layers = BANK LAYER COUNT (what the flag
        #       name says). e.g. 20 layers at interval 2 -> blocks 0,2,...,38 (all
        #       40 DiT blocks covered). Use for new runs.
        n_bank_layers = ((num_layers + interval - 1) // interval) if num_layers_is_span else num_layers
        self.per_layer_proj_out = per_layer_proj_out
        if self.use_proj:
            self.proj_in = nn.Linear(dim, self.inner_dim)
            if per_layer_proj_out:
                # Official Wan2.1-I2V-14B ControlNet layout (controlnet.pth,
                # 0.998B params): ONE shared proj_in but an INDEPENDENT
                # zero-init proj_out per bank layer (20 x Linear(1024, 5120)
                # ~= 105M), so each layer writes its delta into the DiT
                # residual stream through its own matrix instead of one
                # shared projection. State-dict incompatible with the shared
                # layout (proj_out.{i}.weight vs proj_out.weight) — loading
                # fails loudly on flag mismatch.
                self.proj_out = nn.ModuleList([
                    nn.Linear(self.inner_dim, dim) for _ in range(n_bank_layers)])
                for proj in self.proj_out:
                    self._init_proj_out(proj, proj_out_init_std)
            else:
                self.proj_out = nn.Linear(self.inner_dim, dim)
                self._init_proj_out(self.proj_out, proj_out_init_std)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()
        block_dim = self.inner_dim
        if ffn_dim is None:
            ffn_dim = block_dim * 4
        self.interval = interval
        self.start_block = start_block
        self.use_dit_rope = use_dit_rope
        self.layers = nn.ModuleList([
            SingleAttentionBlock(
                dim=block_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                time_embed_dim=time_embed_dim,
            )
            for _ in range(n_bank_layers)
        ])

    @staticmethod
    def _init_proj_out(proj, proj_out_init_std):
        if proj_out_init_std and proj_out_init_std > 0:
            # Non-zero init breaks the zero-init deadlock: with proj_out=0 the upstream
            # attention/ffn/proj_in get ZERO gradient (grad = proj_out.weight^T . grad_res),
            # so the bank never learns. A small std keeps the initial residual gentle
            # while letting gradient flow from step 0. default 0.0 = zero
            # init (no-op start, backward compatible with existing checkpoints).
            nn.init.normal_(proj.weight, mean=0.0, std=proj_out_init_std)
            nn.init.zeros_(proj.bias)
        else:
            nn.init.zeros_(proj.weight)   # zero-init -> residual starts at 0
            nn.init.zeros_(proj.bias)

    def should_apply(self, block_idx):
        if block_idx < self.start_block:
            return False
        offset = block_idx - self.start_block
        return offset % self.interval == 0 and offset // self.interval < len(self.layers)

    def forward_at(self, block_idx, hidden_states, temb, rotary_emb):
        """Returns the RESIDUAL (delta) at `dim`, to be added to hidden_states.

        The layer does ordinary full self-attention over the whole sequence
        ([anchor, video]) with the given rotary_emb — with use_dit_rope, the
        DiT's own full-sequence rope.
        """
        layer_idx = (block_idx - self.start_block) // self.interval
        layer = self.layers[layer_idx]
        x = self.proj_in(hidden_states)          # [B, seq, inner_dim]
        out = layer(x, temb, rotary_emb)
        delta = out - x                           # [B, seq, inner_dim] (just the delta)
        proj = self.proj_out[layer_idx] if self.per_layer_proj_out else self.proj_out
        residual = proj(delta)                    # [B, seq, dim] (delta projected back)
        return residual


class WanI2VChannelAdapter(torch.nn.Module):
    """Native-I2V channel adapter for Wan 14B.

    Takes [native_i2v_input, render_latent, render_mask] (53ch) and maps it
    back to dit.in_dim (36ch) before the original DiT patchify. With
    adapter_type="linear" and mid_channels=0 it is initialized as identity on
    native channels and zero on control channels. adapter_type="residual3d"
    keeps the native channels as a skip path and learns a zero-initialized
    3D-conv delta.
    """

    def __init__(self, in_channels=53, out_channels=36, mid_channels=0, adapter_type="linear", residual_scale=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.adapter_type = adapter_type
        self.residual_scale = residual_scale
        if adapter_type == "linear":
            if mid_channels and mid_channels > 0:
                self.proj = torch.nn.Sequential(
                    torch.nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=True),
                    torch.nn.SiLU(),
                    torch.nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=True),
                )
            else:
                self.proj = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        elif adapter_type == "residual3d":
            hidden = mid_channels if mid_channels and mid_channels > 0 else 256
            self.proj = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, hidden, kernel_size=1, bias=True),
                torch.nn.SiLU(),
                torch.nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=True),
                torch.nn.SiLU(),
                torch.nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=True),
                torch.nn.SiLU(),
                torch.nn.Conv3d(hidden, out_channels, kernel_size=1, bias=True),
            )
        else:
            raise ValueError(f"Unknown I2V adapter type: {adapter_type}")
        self.reset_parameters()

    def reset_parameters(self):
        if self.adapter_type == "linear":
            if isinstance(self.proj, torch.nn.Conv3d):
                torch.nn.init.zeros_(self.proj.weight)
                torch.nn.init.zeros_(self.proj.bias)
                n = min(self.out_channels, self.in_channels)
                with torch.no_grad():
                    for i in range(n):
                        self.proj.weight[i, i, 0, 0, 0] = 1.0
            else:
                for module in self.proj:
                    if isinstance(module, torch.nn.Conv3d):
                        torch.nn.init.zeros_(module.weight)
                        torch.nn.init.zeros_(module.bias)
        else:
            for module in self.proj:
                if isinstance(module, torch.nn.Conv3d):
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                    torch.nn.init.zeros_(module.bias)
            torch.nn.init.zeros_(self.proj[-1].weight)
            torch.nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x):
        if self.adapter_type == "residual3d":
            native = x[:, :self.out_channels]
            return native + self.residual_scale * self.proj(x)
        return self.proj(x)


def model_fn_worldcast(
        dit,
        latents: torch.Tensor,
        timestep: torch.LongTensor,
        context: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        render_latent=None,
        render_mask=None,
        camera_embedding=None,
        camera_proj: Optional[CameraProjection] = None,
        i2v_channel_adapter: Optional[WanI2VChannelAdapter] = None,
        viewpoint_anchor_latent_list: Optional[list] = None,
        viewpoint_anchor_frame_offset: int = 0,
        viewpoint_anchor_mask_list: Optional[list] = None,
        source_self_attn_bank: Optional[SourceSelfAttentionBank] = None,
        anchor_channel_adapter: Optional[AnchorChannelAdapter] = None,
        viewpoint_anchor_traj_offsets: Optional[list] = None,
        # Variable-res (nopad) anchor line: per-view [1,6,1,H_r,W_r] grids in
        # anchor_paths order (WorldCast_prepare_viewpoint_anchors nopad branch).
        viewpoint_anchor_plucker_list: Optional[list] = None,
        anchor_plucker_use_camera_proj: bool = False,
        **kwargs,
) -> torch.Tensor:
    """Anchor-conditioned model function for the frozen Wan2.1-I2V-14B DiT.

    Sequence layout: [anchor tokens (prefix) | video tokens].
    - Video tokens: [latents | y] (36ch) + [render_latent | render_mask] (17ch)
      = 53ch -> WanI2VChannelAdapter -> 36ch -> frozen patch_embedding, plus
      target-camera Pluecker tokens from camera_proj.
    - Anchor tokens (per view, variable resolution): [lat16 | ones4 | lat16 |
      person_mask1] (37ch) -> AnchorChannelAdapter -> 36ch -> frozen
      patch_embedding, plus per-view anchor Pluecker rays (shared camera_proj
      when anchor_plucker_use_camera_proj).
    - SourceSelfAttentionBank residual added to the video tokens only.
    """
    if y is None:
        raise ValueError("model_fn_worldcast requires native Wan I2V `y` condition")
    if i2v_channel_adapter is None:
        raise ValueError("model_fn_worldcast requires an i2v_channel_adapter module")

    target_input = torch.cat([latents.to(dtype), y.to(device=latents.device, dtype=dtype)], dim=1)
    batch_size, _, num_frames, height, width = target_input.shape
    p_t, p_h, p_w = [1, 2, 2]
    f, h, w = num_frames // p_t, height // p_h, width // p_w

    vae_channels = latents.shape[1]
    if render_latent is not None:
        src = render_latent.to(device=target_input.device, dtype=dtype)
        if src.shape[2:] != (num_frames, height, width):
            src = F.interpolate(src, size=(num_frames, height, width), mode="trilinear", align_corners=False)
    else:
        src = torch.zeros(batch_size, vae_channels, num_frames, height, width,
                          device=target_input.device, dtype=dtype)

    if render_mask is not None:
        mask = render_mask.to(device=target_input.device, dtype=dtype)
        mask = F.interpolate(mask, size=(num_frames, height, width), mode="nearest")
    else:
        mask = torch.ones(batch_size, 1, num_frames, height, width,
                          device=target_input.device, dtype=dtype)

    adapter_input = torch.cat([target_input, src, mask], dim=1)
    x = i2v_channel_adapter(adapter_input)
    tokens = dit.patch_embedding(x).flatten(2).transpose(1, 2)

    anchor_len = 0
    anchor_tokens = None
    ntok_per_view = None
    anchor_inputs_list = None
    if viewpoint_anchor_latent_list is not None:
        # Variable-resolution per-view anchors: each view is VAE-encoded at its
        # natural (de-padded) resolution. Patchify each view separately and
        # concatenate along the token dim, so views may differ in latent
        # resolution. Mode "conv": learned AnchorChannelAdapter per view — pack
        # the NATIVE Wan I2V reference layout [lat(16) | ones(4) | lat(16)] with
        # the SAM person mask as ONE extra channel (37ch) and project to
        # dit.in_dim with the identity-init 1x1 Conv3d, then reuse the FROZEN
        # patch_embedding. All tokens are kept — the mask channel lets the
        # model softly gate background instead of hard-dropping tokens.
        if anchor_channel_adapter is None:
            raise ValueError(
                "the conv anchor encoding requires an anchor_channel_adapter module")
        in_dim = getattr(dit, "in_dim", x.shape[1])
        anc_masks = viewpoint_anchor_mask_list or [None] * len(viewpoint_anchor_latent_list)
        tok_list = []
        anchor_inputs_list = []
        # Per-view anchor Pluecker (--use_anchor_plucker + --use_anchor_nopad_mask):
        # one [1,6,1,H_r,W_r] grid per view at that view's own round-up-16
        # resolution, sharing the target camera embedding's normalized frame.
        # --anchor_plucker_use_camera_proj reuses the target's trained
        # camera_proj (zero new parameters; the projection is already trained
        # on exactly this ray distribution).
        anchor_plucker_proj_mod = camera_proj if anchor_plucker_use_camera_proj else None
        ap_list = None
        if viewpoint_anchor_plucker_list is not None:
            if anchor_plucker_proj_mod is None:
                print("[model_fn] WARN: viewpoint_anchor_plucker_list present but no anchor "
                      "Pluecker projection (pass --anchor_plucker_use_camera_proj); "
                      "skipping injection", flush=True)
            else:
                ap_list = list(viewpoint_anchor_plucker_list)
                if len(ap_list) != len(viewpoint_anchor_latent_list):
                    raise ValueError(
                        f"viewpoint_anchor_plucker_list has {len(ap_list)} views but the "
                        f"anchor latents number {len(viewpoint_anchor_latent_list)}")
        for view_index, (anc_lat, anc_mask) in enumerate(zip(viewpoint_anchor_latent_list, anc_masks)):
            anc_lat = anc_lat.to(device=tokens.device, dtype=tokens.dtype)  # [1,z,1,h,w]
            b_a, c_a, _, h_a, w_a = anc_lat.shape
            if anchor_channel_adapter.in_channels == 2 * c_a + 4 + 1:
                # i2v_style base: keep the NATIVE Wan I2V 36ch layout the frozen
                # patch_embedding was pretrained on ([lat|ones4|lat]) and append
                # the SAM person mask as ONE extra channel (37ch), then let the
                # (identity-init) 1x1 conv absorb it back to in_dim.
                if 2 * c_a + 4 != in_dim:
                    raise ValueError(
                        f"anchor_adapter_base i2v_style needs 2*z_dim+4 == "
                        f"dit.in_dim, got 2*{c_a}+4 != {in_dim}")
                m = anc_mask
                if m is None:
                    m = torch.ones(b_a, 1, 1, h_a, w_a,
                                   dtype=anc_lat.dtype, device=anc_lat.device)
                m = m.to(device=anc_lat.device, dtype=anc_lat.dtype)  # [1,1,1,h,w]
                ones4 = torch.ones(b_a, 4, 1, h_a, w_a,
                                   dtype=anc_lat.dtype, device=anc_lat.device)
                anc_input = anchor_channel_adapter(
                    torch.cat([anc_lat, ones4, anc_lat, m], dim=1))
            else:
                raise ValueError(
                    f"anchor_channel_adapter.in_channels "
                    f"({anchor_channel_adapter.in_channels}) does not match the "
                    f"i2v_style no-pad packing 2*z_dim+5 ({2 * c_a + 5}); this "
                    f"release only supports --anchor_adapter_base i2v_style")
            if anc_input.shape[1] != in_dim:
                raise ValueError(
                    f"anchor_channel_adapter output channels ({anc_input.shape[1]}) "
                    f"!= dit.in_dim ({in_dim})")
            anchor_inputs_list.append(anc_input)
            tok_full = dit.patch_embedding(anc_input).flatten(2).transpose(1, 2)  # [1, (h/2)*(w/2), dim]
            if ap_list is not None:
                # Downsample this view's Pluecker grid (its own H_r,W_r) to its
                # latent resolution, project to token space, and ADD — BEFORE
                # anything downstream happens to the anchor tokens, so whatever
                # happens to them happens to the ray-augmented tokens
                # consistently. Flatten order matches dit.patch_embedding
                # (row-major over the (h_a/2, w_a/2) grid).
                ap_i = ap_list[view_index].to(device=tok_full.device, dtype=tok_full.dtype)
                ap_i = F.interpolate(ap_i, size=(1, h_a, w_a), mode="trilinear", align_corners=False)
                ap_tok_i = anchor_plucker_proj_mod(ap_i)  # [1, (h_a/2)*(w_a/2), dim]
                if ap_tok_i.shape[1] != tok_full.shape[1]:
                    raise ValueError(
                        f"anchor plücker view {view_index} produced {ap_tok_i.shape[1]} "
                        f"tokens but the anchor tokens number {tok_full.shape[1]}")
                tok_full = tok_full + ap_tok_i
            tok_list.append(tok_full)
        if os.environ.get("WORLDCAST_DEBUG_ANCHOR"):
            print(f"[nopad-anchor] layout=conv ([lat16|ones4|lat16|person_mask1] -> "
                  f"identity-init AnchorChannelAdapter 37->36 -> frozen patch_embedding, "
                  f"ALL tokens kept) views={len(tok_list)} "
                  f"tokens/view={[t.shape[1] for t in tok_list]}", flush=True)
        anchor_tokens = torch.cat(tok_list, dim=1)  # [1, sum_ntok, dim]
        anchor_len = anchor_tokens.shape[1]
        ntok_per_view = [t.shape[1] for t in tok_list]

    if camera_embedding is not None and camera_proj is not None:
        cam = camera_embedding.to(device=tokens.device, dtype=tokens.dtype)
        cam = F.interpolate(cam, size=(num_frames, height, width), mode="trilinear", align_corners=False).contiguous()
        cam_tokens = camera_proj(cam)
        tokens = tokens + cam_tokens

    temb = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    timestep_proj = dit.time_projection(temb).unflatten(1, (6, -1))
    extra_self_attn_len = anchor_len
    if extra_self_attn_len > 0:
        timestep_proj = timestep_proj.expand(tokens.shape[0], extra_self_attn_len + tokens.shape[1], -1, -1)
    context = dit.text_embedding(context)
    if clip_feature is not None and hasattr(dit, 'img_emb'):
        clip_feature = dit.img_emb(clip_feature)
        context = torch.concat([clip_feature, context], dim=1)

    rotary_emb = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(tokens.device)
    rope_parts = []
    prepend_len = 0
    if anchor_len > 0:
        rope = WanRotaryPosEmbed(128, [1, 2, 2], 1024)
        anchor_offset = viewpoint_anchor_frame_offset if viewpoint_anchor_frame_offset and viewpoint_anchor_frame_offset > 0 else 2 * f
        # Per-view RoPE: each view gets its own temporal offset, and spatial
        # freqs match that view's own latent h,w. viewpoint_anchor_traj_offsets
        # (from the anchor unit) overrides the per-view offsets — the
        # zero-fill-slot line maps each view NAME to a FIXED offset
        # (first_frame=0, front=25, left_side=30, right_side=35, back=40), and
        # the fixed line uses the ordinal offsets frame_offset+vi.
        traj_off = viewpoint_anchor_traj_offsets if viewpoint_anchor_traj_offsets else None
        if traj_off is not None and len(traj_off) != len(anchor_inputs_list):
            traj_off = None  # length mismatch -> legacy offsets
        ropes = []
        for vi, anc_inp in enumerate(anchor_inputs_list):
            r = _build_anchor_rope(
                rope, anc_inp, frame_offset=anchor_offset + vi,
                frame_offsets=[traj_off[vi]] if traj_off is not None else None)
            ropes.append(r)
        anchor_rope = torch.cat(ropes, dim=2)  # [1,1,sum_ntok,head_dim]
        anchor_rope = anchor_rope.squeeze(0).squeeze(0).unsqueeze(1).to(tokens.device)
        rope_parts.append(anchor_rope)
        prepend_len += anchor_len
    if rope_parts:
        rotary_emb = torch.cat(rope_parts + [rotary_emb], dim=0)
        prefix_tokens = []
        if anchor_len > 0:
            prefix_tokens.append(anchor_tokens)
        tokens = torch.cat(prefix_tokens + [tokens], dim=1)

    # Bank rotary: with use_dit_rope, reuse the FULL-sequence rotary_emb built
    # above (token order matches), so the bank attends with the same RoPE the
    # frozen DiT uses — anchors sit at their rope-offset temporal positions
    # (e.g. 25/30/35/40). Requires bank head_dim == DiT rope head_dim.
    bank_rotary = None
    if source_self_attn_bank is not None:
        if getattr(source_self_attn_bank, "use_dit_rope", False):
            _sa = source_self_attn_bank.layers[0]
            _bank_head_dim = _sa.dim // _sa.num_heads
            _rope_head_dim = rotary_emb.shape[-1] * 2
            if _bank_head_dim != _rope_head_dim:
                raise ValueError(
                    f"use_dit_rope: bank head_dim ({_bank_head_dim} = inner_dim "
                    f"{_sa.dim} / heads {_sa.num_heads}) must equal the DiT rope "
                    f"head_dim ({_rope_head_dim}) to share the rope tensor. For 14B "
                    f"(rope head_dim 128) use e.g. --source_self_attn_inner_dim 1024 "
                    f"--source_self_attn_heads 8.")
            bank_rotary = rotary_emb.transpose(0, 1).unsqueeze(1)  # [seq,1,half] -> [1,1,seq,half]
        else:
            raise ValueError(
                "This release only supports source_self_attn_bank with "
                "use_dit_rope=True (the DiT-rope-reuse bank layout); pass "
                "--source_self_attn_use_dit_rope")

    for block_idx, block in enumerate(dit.blocks):
        tokens = block(tokens, context, timestep_proj, rotary_emb)
        # SourceSelfAttentionBank: trainable self-attn residual on [anchor, video]
        # tokens, added only to the video portion (after the prepended prefix).
        if source_self_attn_bank is not None and source_self_attn_bank.should_apply(block_idx):
            sa_residual = source_self_attn_bank.forward_at(
                block_idx, tokens, temb, bank_rotary)
            if os.environ.get("WORLDCAST_DEBUG_BANK_RESIDUAL"):
                _r = sa_residual.float()
                _t = tokens[:, prepend_len:].float() if prepend_len > 0 else tokens.float()
                print(f"[sa-bank-residual] blk{block_idx}: |res|={_r.norm().item():.4f} "
                      f"|tok|={_t.norm().item():.2f} ratio={(_r.norm()/_t.norm()).item():.6f}",
                      flush=True)
            if os.environ.get("WORLDCAST_DISABLE_BANK_RESIDUAL"):
                pass  # debug/ablation: compute but drop the residual (no-op bank)
            elif prepend_len > 0:
                tokens = torch.cat([tokens[:, :prepend_len],
                                    tokens[:, prepend_len:] + sa_residual[:, prepend_len:]], dim=1)
            else:
                tokens = tokens + sa_residual

    if prepend_len > 0:
        tokens = tokens[:, prepend_len:]

    tokens = dit.head(tokens, temb)
    return dit.unpatchify(tokens, (f, h, w))
