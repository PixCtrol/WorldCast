"""WorldCast pipeline: anchor-conditioned, camera-controlled video generation
on the frozen Wan2.1-I2V-14B DiT.

Pipeline units load the reference image / render conditioning / target camera
trajectory, VAE-encode the variable-resolution viewpoint anchors
(no_padding_black variant), and prepare the anchor Pluecker ray grids. The
model function itself (model_fn_worldcast, with the source self-attention
bank, anchor channel adapter and I2V channel adapter) lives in
worldcast_pipeline_interleave.py.
"""
import os
import random
import json
import torch
import numpy as np
from PIL import Image
import einops
import imageio
from tqdm import tqdm
from typing import Optional, Union
from diffusers.video_processor import VideoProcessor

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.controlnet import WanRotaryPosEmbed
from torchvision.transforms import ToTensor
import torch.nn.functional as F


class WorldCastVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.in_iteration_models = ("dit",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WorldCast_Loaddataset(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WorldCast_prepare_latents(),
            WorldCast_prepare_render_latent(),
            WorldCast_prepare_viewpoint_anchors(),
        ]
        # Set by the inference entry point:
        #   pipe.model_fn = model_fn_worldcast  (worldcast_pipeline_interleave)
        self.model_fn = None
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        # Use VAE-specific statistics when available
        if hasattr(self.vae, 'mean') and hasattr(self.vae, 'std'):
            self.vae_latents_mean = self.vae.mean.tolist() if isinstance(self.vae.mean, torch.Tensor) else self.vae.mean
            self.vae_latents_std = self.vae.std.tolist() if isinstance(self.vae.std, torch.Tensor) else self.vae.std
        else:
            self.vae_latents_mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
            self.vae_latents_std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = torch.device("cuda"),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files: bool = True,
        vram_limit: float = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern][0]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]

        # Initialize pipeline
        pipe = WorldCastVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        pipe.dit = model_pool.fetch_model("wan_video_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        # Update VAE-specific settings when the VAE carries its own statistics
        if pipe.vae is not None:
            if hasattr(pipe.vae, 'mean') and hasattr(pipe.vae, 'std'):
                pipe.vae_latents_mean = pipe.vae.mean.tolist() if isinstance(pipe.vae.mean, torch.Tensor) else pipe.vae.mean
                pipe.vae_latents_std = pipe.vae.std.tolist() if isinstance(pipe.vae.std, torch.Tensor) else pipe.vae.std
            if hasattr(pipe.vae, 'upsampling_factor'):
                pipe.vae_scale_factor_spatial = pipe.vae.upsampling_factor
                pipe.video_processor = VideoProcessor(vae_scale_factor=pipe.vae_scale_factor_spatial)
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        reference_image_path: str,
        render_path: str,
        viewpoint_anchors_path: Optional[str] = None,
        viewpoint_anchor_frame_offset: int = 200,
        nframe: int = 81,
        invert_camera_extrinsic: bool = False,
        device: Optional[torch.device] = None,
        negative_prompt: Optional[str] = "",
        dtype: Optional[torch.dtype] = None,
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        end_image: Optional[Image.Image] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        # Scheduler
        num_inference_steps: Optional[int] = 40,
        sigma_shift: Optional[float] = 2.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Camera projection (target trajectory Pluecker -> tokens; also reused
        # for the anchor ray grids via anchor_plucker_use_camera_proj)
        camera_proj=None,
        # Native I2V channel adapter ([latents|y|render|mask] -> original in_dim)
        i2v_channel_adapter=None,
        # Source self-attention bank (residual on [anchor|video] tokens)
        source_self_attn_bank=None,
        # Anchor channel adapter (per-view 37ch packing -> in_dim)
        anchor_channel_adapter=None,
        # Reuse the TARGET's trained camera_proj for the anchor Pluecker rays
        anchor_plucker_use_camera_proj=False,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        if dtype is None:
            dtype = self.torch_dtype
        if device is None:
            device = self.device
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        gen = torch.Generator(device=device)
        # Inputs
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "generator": gen,
            "dtype": dtype,
            "device": device,
            "nframe": nframe,
            "input_image": input_image,
            "render_path": render_path,
            "reference_image_path": reference_image_path,
            "viewpoint_anchors_path": viewpoint_anchors_path,
            "viewpoint_anchor_frame_offset": viewpoint_anchor_frame_offset,
            "invert_camera_extrinsic": invert_camera_extrinsic,
            "end_image": end_image,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "camera_proj": camera_proj,
            "i2v_channel_adapter": i2v_channel_adapter,
            "source_self_attn_bank": source_self_attn_bank,
            "anchor_channel_adapter": anchor_channel_adapter,
            "anchor_plucker_use_camera_proj": anchor_plucker_use_camera_proj,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video

def load_video(video_path):
    """Decode full mp4 to a list of PIL.Image frames.

    Uses decord (in-process libavcodec, batch decode) for speed and reliability.
    Falls back to imageio only if decord raises — keeps the old behavior as a
    last-resort safety net for codec edge cases.
    """
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        # Single batched decode -> ndarray [T, H, W, 3] uint8
        arr = vr.get_batch(range(len(vr))).asnumpy()
        return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
    except Exception:
        reader = imageio.get_reader(video_path)
        try:
            total_frames = reader.count_frames()
            frames = [Image.fromarray(reader.get_data(i)) for i in range(total_frames)]
        finally:
            reader.close()
        return frames


class WorldCast_prepare_latents(PipelineUnit):
    """Native Wan 14B I2V first-frame handling for the channel-adapter path.

    The noisy latents themselves come from WanVideoUnit_InputVideoEmbedder
    (pure noise); this unit only extracts the first-frame latent from the
    native I2V `y` condition so the denoise loop can clamp latent frame 0
    back to it after every scheduler step.
    """

    def __init__(self):
        super().__init__(
            input_params=("y", "device", "dtype"),
            output_params=("first_frame_latents",)
        )

    def process(
        self,
        pipe: WorldCastVideoPipeline,
        y: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> dict:
        if y is None:
            return {}
        # Native Wan 14B I2V y layout is [mask(4), first-frame latent(z_dim)].
        return {"first_frame_latents": y[:, 4:, 0:1].to(device=device, dtype=dtype)}

class WorldCast_prepare_render_latent(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("render_video", "device"),
            output_params=("render_latent")
        )

    def process(self, pipe: WorldCastVideoPipeline, render_video, device):
        render_video = render_video.to(device=device, dtype=pipe.torch_dtype)
        render_latent = pipe.vae.encode(render_video, device)
        return {"render_latent": render_latent}


def _find_nopadblack_aligned_file(anchor_dir, render_path):
    """Locate the aligned JOINT 5-view no_padding_black camera file.

    Render-side first: the per-render file is aligned to THAT render's cam_info
    world frame (several render trees may share one anchor dir while each has
    its own per-run gauge — only the render-side file is in the target gauge).
    The anchor-dir copy assumes a 1:1 anchor-dir <-> render-path mapping and
    stays as the fallback for trees without a render-side file."""
    cands = []
    if render_path:
        cands.append(os.path.join(render_path, "no_padding_black", "anchors_aligned_camera_info.json"))
    if anchor_dir:
        cands.append(os.path.join(anchor_dir, "no_padding_black", "aligned_camera_info.json"))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def _load_target_extrinsics_for_plucker(render_path, num_frames, invert_camera_extrinsic):
    """Target trajectory selection shared with the target camera embedding paths."""
    cam_info = json.load(open(f"{render_path}/cam_info.json", encoding="utf-8"))
    target = torch.tensor(np.array(cam_info["extrinsic"], dtype=np.float32))
    if invert_camera_extrinsic:
        # Old dp3-rendered cam_info is w2c while training data is c2w
        # "as stored". Invert so target poses are c2w before normalization.
        target = torch.linalg.inv(target)
    if target.shape[0] < num_frames:
        raise ValueError(f"cam_info extrinsic has {target.shape[0]} frames < num_frames={num_frames}")
    if target.shape[0] == num_frames:
        frame_indices = list(range(num_frames))
    else:
        frame_indices = np.linspace(0, target.shape[0] - 1, num_frames, dtype=int).tolist()
    return cam_info, target[frame_indices]


def compute_viewpoint_anchor_plucker_nopad(anchor_paths, render_path, num_frames,
                                           sizes, invert_camera_extrinsic=False):
    """Per-view anchor Pluecker for the variable-resolution (no-pad) anchor line.

    Each view gets its own [1, 6, 1, H_r, W_r] grid at that view's natural
    round-up-16 resolution — the resolution its VAE latent (and thus its anchor
    token grid) was encoded at. The normalization frame is still derived from
    the TARGET frames only (target first-view gauge / valid-mean centering /
    scale 2/|C_0|), so anchor rays share the target camera embedding's
    coordinate system; model_fn downsamples each grid to its view's latent
    resolution and adds it to that view's tokens.

    Poses come from the JOINT 5-view no_padding_black estimation aligned to
    the target world. Views are matched BY NAME (stem minus the body_view_
    prefix; the first-frame anchor's stem is already 'first_frame'), so
    anchor-drop subset/permutation sampling stays consistent with the VAE
    latent stack.

    ``sizes``: per-view (H_r, W_r) in anchor_paths order — REQUIRED here; the
    caller (WorldCast_prepare_viewpoint_anchors) derives them from the VAE
    latents it just encoded.

    Returns a list of [1, 6, 1, H_r, W_r] CPU fp32 tensors in anchor_paths
    order, or None when the aligned file is missing / a view has no pose — the
    caller treats None as "no anchor Pluecker for this sample"."""
    body_dirs = [os.path.dirname(p) for p in anchor_paths
                 if os.path.basename(p).startswith("body_view_")]
    anchor_dir = body_dirs[0] if body_dirs else os.path.dirname(anchor_paths[0])
    aligned_path = _find_nopadblack_aligned_file(anchor_dir, render_path)
    if aligned_path is None:
        if os.environ.get("WORLDCAST_DEBUG_ANCHOR"):
            print(f"[anchor-plucker-nopad] no aligned 5-view file for {anchor_dir} / "
                  f"{render_path}; skipping", flush=True)
        return None
    try:
        aligned = json.load(open(aligned_path, encoding="utf-8"))
        views = aligned.get("views") or []
        extr = aligned.get("extrinsic") or []
        if not views or len(views) != len(extr):
            raise ValueError("aligned file views/extrinsic missing or length-mismatched")
        pose_by_name = {name: np.asarray(e, dtype=np.float32) for name, e in zip(views, extr)}
        cam_info, target = _load_target_extrinsics_for_plucker(
            render_path, num_frames, invert_camera_extrinsic)
        from .camera import get_anchor_camera_embedding_shared
        out = []
        for path, size in zip(anchor_paths, sizes):
            stem = os.path.splitext(os.path.basename(path))[0]
            name = stem if stem == "first_frame" else stem.replace("body_view_", "")
            if name not in pose_by_name:
                raise ValueError(f"anchor view {name!r} has no aligned pose (views={views})")
            if size is None:
                raise ValueError("compute_viewpoint_anchor_plucker_nopad requires per-view sizes")
            H_r, W_r = int(size[0]), int(size[1])
            intrinsic = torch.tensor(np.array(cam_info["intrinsic"], dtype=np.float32))
            intrinsic[0, :] = intrinsic[0, :] / cam_info["width"] * W_r
            intrinsic[1, :] = intrinsic[1, :] / cam_info["height"] * H_r
            intrinsic = intrinsic[None]  # [1,3,3] at this view's (H_r, W_r)
            w2c = torch.tensor(pose_by_name[name])   # [4,4] w2c, target world frame
            stored = torch.linalg.inv(w2c)[None]     # [1,4,4] cam_info "as stored" (c2w)
            out.append(get_anchor_camera_embedding_shared(
                intrinsic, stored, 1, target, num_frames, H_r, W_r))
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[anchor-plucker-nopad] WARN: failed for {anchor_dir}: "
              f"{type(e).__name__}: {e}", flush=True)
        return None


# Canonical anchor slot -> FIXED rope temporal offset (used by
# --anchor_random_count_zero_fill). first_frame sits at 0 (its TRUE trajectory
# position); the four body views own the near-axis identity slots 25/30/35/40
# in canonical selected_views order.
ANCHOR_FIXED_SLOTS = {"first_frame": 0, "front": 25, "left_side": 30,
                      "right_side": 35, "back": 40}


class WorldCast_prepare_viewpoint_anchors(PipelineUnit):
    """Load the viewpoint anchor views (front / left_side / right_side / back
    + optional first frame), VAE-encode each at its natural de-padded
    resolution (no_padding_black variant), and emit the per-view latent/mask
    lists consumed by model_fn_worldcast's token-concat prefix branch."""

    ANCHOR_FILENAMES = ("front.jpg", "side.jpg", "back.jpg")

    def __init__(self):
        super().__init__(
            input_params=("viewpoint_anchors_path", "height", "width", "device",
                          "render_path", "invert_camera_extrinsic", "viewpoint_anchor_frame_offset",
                          # Anchor Pluecker (--use_anchor_plucker): needed to reproduce
                          # the target camera embedding's frame selection.
                          "num_frames"),
            output_params=("viewpoint_anchor_latent_list", "viewpoint_anchor_mask_list",
                           "viewpoint_anchor_traj_offsets", "viewpoint_anchor_plucker_list"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: "WorldCastVideoPipeline", viewpoint_anchors_path, height, width,
                device, render_path=None, invert_camera_extrinsic=None,
                viewpoint_anchor_frame_offset=None, num_frames=None):
        if not viewpoint_anchors_path:
            return {}
        if not getattr(pipe, "use_anchor_nopad_mask", False):
            raise ValueError(
                "This release only implements the variable-resolution anchor path; "
                "pass --use_anchor_nopad_mask (with --use_anchor_no_padding_black)")

        # Support both directory paths and semicolon-separated file paths.
        if os.path.isdir(viewpoint_anchors_path):
            anchor_paths = [os.path.join(viewpoint_anchors_path, fname) for fname in self.ANCHOR_FILENAMES]
            anchor_paths = [path for path in anchor_paths if os.path.isfile(path)]
            if not anchor_paths:
                # Anchor directories store arbitrary *.png names (body_view_*.png).
                # pipe.anchor_max_frames caps how many sorted PNGs are loaded
                # (default 4).
                max_n = getattr(pipe, "anchor_max_frames", 4)
                anchor_paths = [
                    os.path.join(viewpoint_anchors_path, fname)
                    for fname in sorted(os.listdir(viewpoint_anchors_path))
                    if fname.lower().endswith(".png")
                ]
                if max_n and max_n > 0:
                    anchor_paths = anchor_paths[:max_n]
        elif ";" in viewpoint_anchors_path:
            # Semicolon-separated file paths (e.g. body_view_front.png;body_view_left_side.png)
            anchor_paths = [p.strip() for p in viewpoint_anchors_path.split(";") if p.strip()]
            anchor_paths = [p for p in anchor_paths if os.path.isfile(p)]
        else:
            return {}

        if not anchor_paths:
            return {}
        # Semantic-view order: reorder anchor_paths to body_views_meta.json ->
        # selected_views (front, left_side, right_side, back) so view index maps
        # deterministically to a semantic view. Enabled by
        # --use_anchor_selected_views_order.
        if getattr(pipe, "use_anchor_selected_views_order", False):
            anchor_paths = self._order_by_selected_views(anchor_paths, viewpoint_anchors_path)
        # A1 arbitrary-anchor-count mode (--use_anchor_random_count /
        # --use_first_frame_anchor) generates offsets PER SAMPLE inside the
        # nopad branch below ([0 for the first-frame anchor] + ordinal 25+5i
        # for the sampled body views).
        a1_mode = bool(getattr(pipe, "use_anchor_random_count", False)
                       or getattr(pipe, "use_first_frame_anchor", False))
        traj_offsets = []  # canonical "not provided" marker for model_fn
        # VGGT alignment: re-resolve anchor paths through the SAME shared
        # semantic resolver used by the VGGT token paths, so the VAE anchor
        # tokens and any other per-view consumer can never diverge in view
        # order (front -> left_side -> right_side -> back) even if
        # --use_anchor_selected_views_order was omitted. Falls back to the
        # paths resolved above for non-semantic dirs / explicit file lists.
        if (os.path.isdir(viewpoint_anchors_path)
                and os.path.isfile(os.path.join(viewpoint_anchors_path, "body_views_meta.json"))):
            # Guard on body_views_meta.json so legacy front/side/back or
            # arbitrary-name dirs keep their original reading precedence.
            try:
                from scripts.data.anchor_views import resolve_semantic_anchor_paths
                semantic_paths, _view_names = resolve_semantic_anchor_paths(
                    viewpoint_anchors_path,
                    prefer_nopad=False,
                    max_views=getattr(pipe, "anchor_max_frames", 4),
                )
                if semantic_paths:
                    anchor_paths = semantic_paths
            except ImportError:
                pass
        # First-frame anchor (--use_first_frame_anchor): prepend the
        # first-frame person crop <render_path>/no_padding_black/first_frame.png
        # (natural tight resolution, black background) as an anchor at rope
        # offset 0 — its TRUE trajectory position (the first frame IS latent
        # frame 0), so early video frames are rope-nearest to it. Skipped
        # silently when absent — those samples just run with body views only.
        first_frame_added = False
        if getattr(pipe, "use_first_frame_anchor", False) and render_path:
            ff_path = os.path.join(render_path, "no_padding_black", "first_frame.png")
            if os.path.isfile(ff_path):
                anchor_paths = [ff_path] + anchor_paths
                first_frame_added = True
        # A1 arbitrary anchor count (--use_anchor_random_count): per-sample
        # random SUBSET of the body views. With
        # --anchor_random_count_no_shuffle the body views keep CANONICAL
        # selected_views order (first k) — random-count DROP without
        # permutation. The first-frame anchor is never part of the sampling
        # pool.
        # --anchor_random_count_zero_fill requires the k sampler; without it
        # the flag would be a silent no-op.
        _zero_fill_idx = None
        if getattr(pipe, "anchor_random_count_zero_fill", False):
            if not getattr(pipe, "use_anchor_random_count", False):
                raise ValueError(
                    "--anchor_random_count_zero_fill requires --use_anchor_random_count "
                    "(it only changes HOW unsampled slots are represented: zero-fill "
                    "placeholder tokens instead of sequence gaps)")
        if getattr(pipe, "use_anchor_random_count", False):
            body = anchor_paths[1:] if first_frame_added else anchor_paths
            # --anchor_random_count_priority: comma-separated view names
            # reordering `body` by priority BEFORE the k-trim, so no_shuffle's
            # body[:k] keeps the listed views first. E.g.
            # "back,front,left_side,right_side": k=1 -> [back], k=2 ->
            # [back, front], k=3 -> [back, front, left_side], k=4 -> all.
            # Unlisted views keep their canonical relative order after the
            # listed ones (sorted is stable). Typos are a hard error — a
            # misspelled priority would silently change WHICH slots are real.
            # In zero-fill mode only WHICH slots are real changes (offsets
            # stay name-mapped); in plain A1 mode the ordinal 25+5i offsets
            # follow the reordered sequence.
            _prio = getattr(pipe, "anchor_random_count_priority", None)
            if _prio:
                def _body_view_name(p):
                    _stem = os.path.splitext(os.path.basename(p))[0]
                    return _stem if _stem == "first_frame" else _stem.replace("body_view_", "")
                _unknown = [n for n in _prio if n not in ANCHOR_FIXED_SLOTS]
                if _unknown:
                    raise ValueError(
                        f"anchor_random_count_priority has unknown view name(s) "
                        f"{_unknown}; expected names from {sorted(ANCHOR_FIXED_SLOTS)}")
                _rank = {n: i for i, n in enumerate(_prio)}
                body = sorted(body, key=lambda p: _rank.get(_body_view_name(p), len(_rank)))
            k_min_cfg = max(0, int(getattr(pipe, "anchor_random_count_min", 1)))
            k_max_cfg = int(getattr(pipe, "anchor_random_count_max", 4))
            lo, hi = min(k_min_cfg, len(body)), min(k_max_cfg, len(body))
            if hi < lo:
                hi = lo
            k = random.randint(lo, hi)
            if k == 0 and not first_frame_added:
                # k=0 is only meaningful with a first-frame anchor; an empty
                # anchor set would crash model_fn's token concat.
                k = 1
            if getattr(pipe, "anchor_random_count_no_shuffle", False):
                sampled = list(body[:k])
            else:
                sampled = list(random.sample(body, k))
            # --anchor_random_count_zero_fill: keep the FULL canonical slot
            # structure on every sample — every body view stays in
            # anchor_paths at its FIXED slot offset, and the k unsampled
            # views become ZERO-FILL anchors (all-zero image at the missing
            # view's own natural resolution + all-zero person mask, encoded
            # through the SAME per-view chain). The sequence always carries
            # slots 0/25/30/35/40; an empty slot is an explicit zero-content
            # token, not a rope gap. The k sampling above only decides WHICH
            # slots are real — permutation is irrelevant because each view is
            # restored to its own slot.
            if getattr(pipe, "anchor_random_count_zero_fill", False):
                _zf_names = {os.path.splitext(os.path.basename(p))[0] for p in body} \
                    - {os.path.splitext(os.path.basename(p))[0] for p in sampled}
                anchor_paths = ([anchor_paths[0]] if first_frame_added else []) + list(body)
                _zero_fill_idx = {i for i, p in enumerate(anchor_paths)
                                  if os.path.splitext(os.path.basename(p))[0] in _zf_names}
                # Fixed offsets BY NAME (slot identity must not depend on the
                # meta's selected_views order). Unknown names are a hard
                # error — a guessed offset would corrupt the slot structure
                # this variant exists to keep constant.
                _zf_offs = []
                for _p in anchor_paths:
                    _stem = os.path.splitext(os.path.basename(_p))[0]
                    _name = _stem if _stem == "first_frame" else _stem.replace("body_view_", "")
                    if _name not in ANCHOR_FIXED_SLOTS:
                        raise ValueError(
                            f"[anchor-zero-fill] unknown anchor view name {_stem!r} "
                            f"({os.path.basename(_p)}); fixed slots need one of "
                            f"{sorted(ANCHOR_FIXED_SLOTS)}")
                    _zf_offs.append(ANCHOR_FIXED_SLOTS[_name])
                traj_offsets = _zf_offs
                if not first_frame_added:
                    print("[anchor-zero-fill] WARN: first-frame anchor absent for "
                          f"{render_path} — zero-filling the {len(_zf_names)} unsampled "
                          "body slots only (no offset-0 slot this sample)", flush=True)
                if os.environ.get("WORLDCAST_DEBUG_ANCHOR"):
                    _vn = [(os.path.splitext(os.path.basename(_p))[0]
                            .replace("body_view_", ""),
                            "ZF" if _i in _zero_fill_idx else "real")
                           for _i, _p in enumerate(anchor_paths)]
                    print(f"[anchor-zero-fill] k_body={len(body) - len(_zf_names)} "
                          f"slots={_vn} offsets={_zf_offs}", flush=True)
            else:
                anchor_paths = ([anchor_paths[0]] if first_frame_added else []) + sampled
        # Per-sample offsets for A1 mode: first-frame anchor at 0, body views
        # at ordinal 25+5i (n=4 body views reproduces the legacy 25,30,35,40).
        # Zero-fill already assigned NAME-mapped slot offsets above (slot
        # identity must not depend on the meta's selected_views order) — keep
        # them.
        if a1_mode and _zero_fill_idx is None:
            n_ff = 1 if first_frame_added else 0
            traj_offsets = ([0] if n_ff else []) + \
                [25 + 5 * i for i in range(len(anchor_paths) - n_ff)]
            if os.environ.get("WORLDCAST_DEBUG_ANCHOR"):
                print(f"[anchor-a1] first_frame={'yes' if n_ff else 'no'} "
                      f"k_body={len(anchor_paths) - n_ff} "
                      f"views={[os.path.splitext(os.path.basename(p))[0] for p in anchor_paths]} "
                      f"offsets={traj_offsets}", flush=True)
        result = self._encode_per_view_nopad(
            pipe, anchor_paths, device, zero_fill_indices=_zero_fill_idx)
        # Anchor Pluecker for the variable-res nopad line (--use_anchor_plucker
        # + --use_anchor_nopad_mask): one [1,6,1,H_r,W_r] grid per view at that
        # view's OWN round-up-16 resolution (H_r = latent_h * 8), in the target
        # camera embedding's normalized frame, from the JOINT 5-view
        # no_padding_black estimation aligned to the target world (anchor-dir
        # file for train, render-dir file for benchmark). model_fn adds each
        # view's projected tokens to that view's anchor tokens.
        if getattr(pipe, "use_anchor_plucker", False) and render_path and num_frames:
            lat_list = result.get("viewpoint_anchor_latent_list") or []
            if lat_list:
                sizes = [(l.shape[-2] * 8, l.shape[-1] * 8) for l in lat_list]
                plucker_list = compute_viewpoint_anchor_plucker_nopad(
                    anchor_paths, render_path, num_frames, sizes=sizes,
                    invert_camera_extrinsic=bool(invert_camera_extrinsic))
                if plucker_list is not None:
                    # zero-fill slots carry a zero-image latent + zero mask —
                    # their ray grids must be ZEROED TOO, otherwise the empty
                    # slot leaks the missing view's true camera geometry from
                    # the aligned 5-view file and the anchor-count ablation is
                    # confounded. A zero grid means zero additive contribution
                    # after camera_proj — consistent with "the anchor does not
                    # exist", matching the zero image/mask.
                    if _zero_fill_idx:
                        for _i in _zero_fill_idx:
                            if _i < len(plucker_list):
                                plucker_list[_i] = torch.zeros_like(plucker_list[_i])
                        if os.environ.get("WORLDCAST_DEBUG_ANCHOR"):
                            _zf = sorted(_zero_fill_idx & set(range(len(plucker_list))))
                            print(f"[anchor-plucker-nopad] zero-filled ray grids at "
                                  f"slots {_zf} ({len(plucker_list)} views total)",
                                  flush=True)
                    if os.environ.get("WORLDCAST_DEBUG_ANCHOR"):
                        print(f"[anchor-plucker-nopad] {len(plucker_list)} views "
                              f"sizes={[(t.shape[-2], t.shape[-1]) for t in plucker_list]}",
                              flush=True)
                    result["viewpoint_anchor_plucker_list"] = plucker_list
        result["viewpoint_anchor_traj_offsets"] = traj_offsets
        return result

    def _order_by_selected_views(self, anchor_paths, viewpoint_anchors_path):
        """Reorder anchor_paths to the semantic order recorded in
        body_views_meta.json -> selected_views (e.g. front, left_side, right_side,
        back). The view name is the file stem with the "body_view_" prefix stripped
        (body_view_front.png -> "front"). Falls back to the input order when the
        meta file is missing/unparseable or contains no selected_views."""
        meta_path = (os.path.join(viewpoint_anchors_path, "body_views_meta.json")
                     if isinstance(viewpoint_anchors_path, str)
                     and os.path.isdir(viewpoint_anchors_path) else None)
        if not meta_path or not os.path.isfile(meta_path):
            return anchor_paths
        try:
            selected = json.load(open(meta_path)).get("selected_views") or []
        except Exception:
            return anchor_paths
        if not selected:
            return anchor_paths
        stem2path = {}
        for p in anchor_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            stem2path[stem.replace("body_view_", "")] = p
        ordered = [stem2path[name] for name in selected if name in stem2path]
        selected_set = set(selected)
        rest = [p for p in anchor_paths
                if os.path.splitext(os.path.basename(p))[0].replace("body_view_", "")
                not in selected_set]
        return ordered + rest

    def _encode_per_view_nopad(self, pipe, anchor_paths, device, zero_fill_indices=None):
        """Per-view VAE at natural (de-padded) resolution + person-mask channel.

        Each anchor view is loaded from its ``no_padding_black/`` sibling
        (person kept, background zeroed; falling back to ``nopad/`` then the
        padded original), rounded up to a multiple of 16 (VAE 8x + patch 2x),
        and VAE-encoded INDEPENDENTLY -> [1,z,1,h_i,w_i]. The SAM3 person mask
        (from the ``person_mask/`` sibling) is resized to the latent resolution
        (h_i,w_i) as a single [1,1,1,h_i,w_i] channel.

        Returns LIST outputs (``viewpoint_anchor_latent_list`` /
        ``viewpoint_anchor_mask_list``) because each view may have a different
        spatial resolution — the model_fn consumes them by token-dim
        concatenation instead of a uniform [B,C,n_anc,h,w] stack.
        """
        pipe.load_models_to_device(self.onload_model_names)
        use_black = getattr(pipe, "use_anchor_no_padding_black", False)
        lat_list = []
        mask_list = []
        for i, path in enumerate(anchor_paths):
            d = os.path.dirname(path)
            fname = os.path.basename(path)
            nopad_path = path
            # --anchor_random_count_zero_fill slot (zero_fill_indices): encode an
            # ALL-ZERO anchor at the missing view's OWN natural resolution (read
            # from the same variant-chain file the real view would use — token
            # count per slot stays IDENTICAL to the full-anchor case, so the
            # model cannot use sequence length as an emptiness cue) with an
            # all-zero person mask (explicit "no person", not the mask-missing
            # default of all-ones). Zero latents are deterministic per
            # resolution — cache them on the pipe.
            if zero_fill_indices and i in zero_fill_indices:
                _zf_subs = (("no_padding_black",) if use_black else ()) + ("nopad",)
                _zf_file = path
                for sub in _zf_subs:
                    cand = os.path.join(d, sub, fname)
                    if os.path.isfile(cand):
                        _zf_file = cand
                        break
                with Image.open(_zf_file) as _im:
                    W, H = _im.size
                H_r = ((H + 15) // 16) * 16
                W_r = ((W + 15) // 16) * 16
                cache = getattr(pipe, "_zero_fill_latent_cache", None)
                if cache is None:
                    cache = pipe._zero_fill_latent_cache = {}
                ck = (H_r, W_r, str(pipe.torch_dtype))
                if ck not in cache:
                    _z = torch.zeros(1, 3, 1, H_r, W_r, dtype=pipe.torch_dtype, device=pipe.device)
                    cache[ck] = pipe.vae.encode(_z, device=pipe.device).to(dtype=pipe.torch_dtype)
                lat_list.append(cache[ck].to(device=pipe.device))
                h, w = cache[ck].shape[-2], cache[ck].shape[-1]
                mask_list.append(torch.zeros(1, 1, 1, h, w, dtype=pipe.torch_dtype,
                                             device=pipe.device))
                continue
            subs = (("no_padding_black",) if use_black else ()) + ("nopad",)
            for sub in subs:
                cand = os.path.join(d, sub, fname)
                if os.path.isfile(cand):
                    nopad_path = cand
                    break
            img = Image.open(nopad_path).convert("RGB")
            W, H = img.size
            # Round up to a multiple of 16 so the (8x VAE + 2x patch) chain lands on
            # integer latent/patch dims. Minimal (<=15px) resize, no black-canvas pad.
            H_r = ((H + 15) // 16) * 16
            W_r = ((W + 15) // 16) * 16
            img_t = pipe.video_processor.preprocess(img, height=H_r, width=W_r)
            img_t = img_t.to(device=pipe.device, dtype=torch.float32).unsqueeze(2)  # [1,C,1,H,W]
            img_t = img_t.to(dtype=pipe.torch_dtype)
            lat = pipe.vae.encode(img_t, device=pipe.device)  # [1,z,1,h,w]
            lat_list.append(lat.to(dtype=pipe.torch_dtype, device=pipe.device))
            # Mask lookup follows the RESOLVED view dir first, then falls back
            # to the anchor-root person_mask/ (no_padding_black views share the
            # root mask — same resolution, no crop).
            mask_path = os.path.join(os.path.dirname(nopad_path), "person_mask", fname)
            if not os.path.isfile(mask_path):
                mask_path = os.path.join(d, "person_mask", fname)
            if os.path.isfile(mask_path):
                mask_np = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(np.float32)  # [H,W]
                h, w = lat.shape[-2], lat.shape[-1]
                mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                mask_t = F.interpolate(mask_t, size=(h, w), mode="nearest")  # [1,1,h,w]
                mask_t = mask_t.unsqueeze(2)  # [1,1,1,h,w]
                mask_list.append(mask_t.to(dtype=pipe.torch_dtype, device=pipe.device))
            else:
                mask_list.append(None)
        if os.environ.get("WORLDCAST_DEBUG_ANCHOR"):
            _dbg = []
            for _i, (_p, _l, _m) in enumerate(zip(anchor_paths, lat_list, mask_list)):
                _st = os.path.splitext(os.path.basename(_p))[0].replace("body_view_", "")
                _zf = "~zf" if (zero_fill_indices is not None and _i in zero_fill_indices) else ""
                _dbg.append(f"{_st}{_zf}[{tuple(_l.shape)}:mask={_m is not None}]")
            _src = "no_padding_black" if use_black else "nopad"
            print(f"[nopad-anchor] src={_src} {len(lat_list)} views: " + "  ".join(_dbg), flush=True)
        return {
            "viewpoint_anchor_latent_list": lat_list,
            "viewpoint_anchor_mask_list": mask_list,
        }


def cast_to_dtype_device(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return x

class WorldCast_Loaddataset(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image_path", "render_path", "input_video", "nframe", "invert_camera_extrinsic", "device", "height", "width"),
            output_params=("input_image", "render_video", "render_mask", "camera_embedding", "height", "width")
        )

    def process(self, pipe: WorldCastVideoPipeline, reference_image_path, render_path, input_video, nframe,
                 device, height=None, width=None, invert_camera_extrinsic=False):
        from .camera import get_camera_embedding
        if isinstance(reference_image_path, Image.Image):
            image = reference_image_path.convert("RGB")
        elif isinstance(reference_image_path, (str, os.PathLike)) and str(reference_image_path):
            image = Image.open(reference_image_path).convert("RGB")
        elif input_video is not None and len(input_video) > 0:
            image = input_video[0].convert("RGB") if isinstance(input_video[0], Image.Image) else Image.open(input_video[0]).convert("RGB")
        else:
            raise ValueError("reference_image_path is empty and input_video is unavailable; cannot build the reference image")

        # Use pipeline-provided height/width if available, otherwise compute from image aspect ratio
        if height is not None and width is not None:
            pass  # use the provided values directly
        else:
            h, w = image.height, image.width
            if h >= w:
                height = 1024
                width = 768
            else:
                height = 768
                width = 1024

        image = image.resize((width, height))

        # load conditions
        render_frames = load_video(f"{render_path}/render.mp4")
        render_width, render_height = render_frames[0].size
        if len(render_frames) < nframe:
            raise ValueError(f"render.mp4 has {len(render_frames)} frames, but nframe={nframe}")
        if len(render_frames) == nframe:
            frame_indices = list(range(nframe))
        else:
            # Match GT LoadVideo operator: np.linspace(..., dtype=int) uses floor truncation
            frame_indices = np.linspace(0, len(render_frames) - 1, nframe, dtype=int).tolist()
        render_frames = [render_frames[i] for i in frame_indices]
        render_video = torch.stack([ToTensor()(frame) for frame in render_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
        render_video = F.interpolate(render_video, size=(height, width), mode='bicubic')[None]
        # replace the first frame with the high-quality reference image
        render_video[0, 0] = ToTensor()(image) * 2 - 1
        render_video = torch.clip(render_video, -1, 1)  # [-1~1], [1,f,c,h,w]
        render_mask_frames = load_video(f"{render_path}/render_mask.mp4")
        if len(render_mask_frames) < nframe:
            raise ValueError(f"render_mask.mp4 has {len(render_mask_frames)} frames, but nframe={nframe}")
        render_mask_frames = [render_mask_frames[i] for i in frame_indices]
        render_mask = torch.stack([ToTensor()(frame) for frame in render_mask_frames], dim=0)[:, 0:1]  # [f,1,h,w]
        render_mask = F.interpolate(render_mask, size=(height, width), mode='nearest')[None]  # [0,1],[1,f,1,h,w]
        render_video = einops.rearrange(render_video, "b f c h w -> b c f h w")
        render_mask = einops.rearrange(render_mask, "b f c h w -> b c f h w")
        render_mask[render_mask < 0.5] = 0
        render_mask[render_mask >= 0.5] = 1

        # load camera
        cam_info = json.load(open(f"{render_path}/cam_info.json"))
        w2cs = torch.tensor(np.array(cam_info["extrinsic"]), dtype=torch.float32, device=device)
        if w2cs.shape[0] < nframe:
            raise ValueError(f"cam_info extrinsic has {w2cs.shape[0]} frames, but nframe={nframe}")
        w2cs = w2cs[frame_indices]
        # The training data stores cam_info["extrinsic"] as c2w, while old dp3
        # benchmark data stores it as w2c. get_camera_embedding() treats its
        # input as w2c (it computes c2w = inv(extrinsic)), so to make a
        # w2c-stored benchmark match the c2w-stored training distribution we
        # must pre-invert here.
        if invert_camera_extrinsic:
            w2cs = torch.inverse(w2cs)
        intrinsic = torch.tensor(np.array(cam_info["intrinsic"]), dtype=torch.float32, device=device)
        cam_width = cam_info.get("width", render_width)
        cam_height = cam_info.get("height", render_height)
        intrinsic[0, :] = intrinsic[0, :] / cam_width * width
        intrinsic[1, :] = intrinsic[1, :] / cam_height * height
        intrinsic = intrinsic[None].repeat(nframe, 1, 1)
        camera_embedding = get_camera_embedding(intrinsic, w2cs, nframe, height, width, normalize=True)

        device = pipe.device
        dtype = pipe.torch_dtype
        render_video = cast_to_dtype_device(render_video, device, dtype)
        render_mask = cast_to_dtype_device(render_mask, device, dtype)
        camera_embedding = cast_to_dtype_device(camera_embedding, device, dtype)
        return {"input_image": image, "render_video": render_video, "render_mask": render_mask, "camera_embedding": camera_embedding, "height": height, "width": width, "image": image, "num_channels_latents": pipe.vae.z_dim, "batch_size": 1}


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WorldCastVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",)
        )

    def process(self, pipe: WorldCastVideoPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}



class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WorldCastVideoPipeline, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(self, pipe: WorldCastVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WorldCastVideoPipeline, prompt) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WorldCastVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((224, 224))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}



class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WorldCastVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


def _build_anchor_rope(rope_module, anchor_5d, frame_offset, frame_offsets=None):
    """Build WanRotaryPosEmbed-shaped freqs for an anchor latent with a temporal offset.

    Returns a tensor of shape [1, 1, ppf*pph*ppw, head_dim] matching `WanRotaryPosEmbed.forward`.
    When `frame_offsets` (list of ppf ints) is given, it REPLACES the contiguous
    arange(frame_offset, frame_offset+ppf) temporal indices — used by the
    zero-fill-slot / A1 anchor lines to place each per-view anchor latent
    frame at its own fixed slot's temporal position.
    """
    _, _, num_frames, height, width = anchor_5d.shape
    p_t, p_h, p_w = rope_module.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
    rope_module.freqs = rope_module.freqs.to(anchor_5d.device)
    head_dim = rope_module.attention_head_dim
    freqs = rope_module.freqs.split_with_sizes(
        [head_dim // 2 - 2 * (head_dim // 6), head_dim // 6, head_dim // 6], dim=1
    )
    if frame_offsets is not None and len(frame_offsets) == ppf:
        f_idx = torch.as_tensor(list(frame_offsets), dtype=torch.long, device=anchor_5d.device)
    else:
        f_idx = torch.arange(frame_offset, frame_offset + ppf, device=anchor_5d.device)
    freqs_f = freqs[0][f_idx].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
    return torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
