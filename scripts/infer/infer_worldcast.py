#!/usr/bin/env python3
"""
WorldCast inference — anchor-conditioned, camera-controlled video generation
on the frozen Wan2.1-I2V-14B-720P DiT.

Architecture (fixed; flags below must match the released checkpoint):
  - native I2V channel adapter: [latents|y|render|mask] 53ch -> 36ch -> frozen
    patch_embedding
  - viewpoint anchors as a token-concat prefix, variable-resolution
    no_padding_black encoding through the anchor channel adapter (37ch -> 36ch)
  - per-view anchor Pluecker rays reusing the (frozen) target camera_proj
  - source self-attention bank over [anchor|video] tokens, residual added to
    the video tokens

Usage:
  python scripts/infer/infer_worldcast.py --ckpt ... --json manifest.json
"""
import argparse, csv, json, os, shutil, sys, torch
from PIL import Image

# Make `diffsynth` importable regardless of CWD (script lives in scripts/infer/).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from safetensors.torch import load_file
from diffsynth.pipelines.worldcast_pipeline import WorldCastVideoPipeline, ModelConfig
from diffsynth.pipelines.worldcast_pipeline_interleave import (
    model_fn_worldcast, CameraProjection, WanI2VChannelAdapter,
    SourceSelfAttentionBank, AnchorChannelAdapter,
)
from diffsynth.utils.data import save_video
from diffsynth.core.data.operators import LoadVideo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/worldcast_bank.safetensors",
                   help="Trained bank checkpoint (source_self_attn_bank.* + anchor_channel_adapter.*).")
    p.add_argument("--init_dit_from", default=None,
                   help="Frozen base DiT checkpoint the bank was trained against. "
                        "Provides the frozen DiT weights + camera_proj + i2v_channel_adapter. "
                        "Required: bank-only checkpoints do not contain the frozen base.")
    p.add_argument("--csv", default="")
    p.add_argument("--json", default=None,
                   help="Optional JSON manifest (plain list of objects with the CSV field names: prompt, "
                        "reference_image_path, render_path, optional viewpoint_anchors_path etc). "
                        "Takes precedence over --csv when set; sharding identical to CSV.")
    p.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate (per shard if --num_shards > 1)")
    p.add_argument("--shard_index", type=int, default=0, help="Shard index for multi-GPU parallel inference (0-based)")
    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards for parallel inference")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--sigma_shift", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_suffix", default="", help="Optional suffix appended before .mp4 in output filenames")
    p.add_argument("--save_render_concat", action="store_true", help="Also save a side-by-side render|generated comparison video")
    p.add_argument("--copy_conditioning", action="store_true", help="Copy reference/render/anchor conditioning into <output_dir>/<sample>/ for inspection")
    p.add_argument("--traj", default="",
                   help="build + generate named target trajectories per case instead of "
                        "the base render trajectory: comma-separated presets from "
                        "scripts/data/trajectories.py — orbit_360, lemniscate, "
                        "orbit_tilt_20, orbit_180_push, zoom_out, turn_left, "
                        "turn_right, zoom_in, or orbit_<N>[_cw|_ccw] (N=1..360 "
                        "horizontal arc, cw/ccw viewed from above). Each row's "
                        "render_path "
                        "is used as the BASE case dir (needs cam_info.json c2w + "
                        "depth/first_frame_depth.npy + first_frame.png); the row is "
                        "replaced by one row per trajectory, outputs "
                        "<case>__<preset>_seed<seed>.mp4")
    p.add_argument("--traj_out_root", default="",
                   help="root dir for the generated trajectory conditioning dirs "
                        "(default: sibling of each render_path)")
    p.add_argument("--traj_nframes", type=int, default=0,
                   help="trajectory length in frames (default: --num_frames)")
    p.add_argument("--model_root", default=None,
                   help="Wan2.1-I2V-14B-720P base weights root (default: pretrained/Wan2.1-I2V-14B-720P).")
    p.add_argument("--invert_camera_extrinsic", action="store_true",
                   help="Invert cam_info extrinsic (w2c->c2w) before computing the camera embedding. "
                   "Use for old dp3-rendered benchmark cases whose cam_info stores w2c while training stored c2w.")
    # --- source self-attention bank (must match the checkpoint's training config) ---
    p.add_argument("--use_source_self_attn", action="store_true",
                   help="Inject anchor+target via SourceSelfAttentionBank (trainable self-attn residual). "
                   "Required — the released checkpoints are bank-only.")
    p.add_argument("--source_self_attn_heads", type=int, default=24)
    p.add_argument("--source_self_attn_layers", type=int, default=30)
    p.add_argument("--source_self_attn_interval", type=int, default=2)
    p.add_argument("--source_self_attn_start_block", type=int, default=0,
                   help="Must match training: first DiT block receiving source self-attn residual.")
    p.add_argument("--source_self_attn_inner_dim", type=int, default=1024)
    p.add_argument("--source_self_attn_ffn_mult", type=float, default=4.0)
    p.add_argument("--source_self_attn_per_layer_proj_out", action="store_true",
                   help="MUST match training: independent proj_out per bank layer (official Wan "
                   "ControlNet layout, proj_out.{i}.* keys). Mismatch with the checkpoint "
                   "fails loudly at load_state_dict.")
    p.add_argument("--source_self_attn_layers_are_bank_layers", action="store_true",
                   help="--source_self_attn_layers counts BANK layers instead of the legacy "
                   "DiT-block span. Legacy (default): 20/interval2 -> 10 layers at blocks "
                   "0-18. With flag: 20/interval2 -> 20 layers at blocks 0-38. "
                   "Must match training.")
    p.add_argument("--source_self_attn_use_dit_rope", action="store_true",
                   help="SourceSelfAttentionBank reuses the DiT's full-sequence rotary_emb tensor "
                   "(anchors at their slot offsets) instead of building its own "
                   "fresh 2D rope. Requires bank head_dim == DiT rope head_dim (128 for 14B -> "
                   "--source_self_attn_inner_dim 1024 --source_self_attn_heads 8). "
                   "Must match training.")
    # --- viewpoint anchors ---
    p.add_argument("--use_viewpoint_anchors", action="store_true",
                   help="Read viewpoint_anchors_path from the manifest and inject anchors as a "
                   "token-concat prefix. Required — WorldCast is anchor-conditioned.")
    p.add_argument("--viewpoint_anchor_frame_offset", type=int, default=200)
    p.add_argument("--anchor_max_frames", type=int, default=4,
                   help="Maximum sorted anchor frames to load; 0 means all frames. Must match training.")
    p.add_argument("--use_anchor_selected_views_order", action="store_true",
                   help="Order anchor views by body_views_meta.json -> selected_views (semantic "
                   "order) instead of sorted filename order. Must match training.")
    p.add_argument("--use_anchor_nopad_mask", action="store_true",
                   help="Per-view VAE at natural (de-padded) resolution + person mask channel. "
                   "Required — the variable-resolution anchor path is the only implemented one.")
    p.add_argument("--use_anchor_no_padding_black", action="store_true",
                   help="With --use_anchor_nopad_mask: read each view from its no_padding_black/ "
                   "sibling (person kept, background zeroed) instead of nopad/. Must match "
                   "training.")
    p.add_argument("--use_first_frame_anchor", action="store_true",
                   help="Prepend <render_path>/no_padding_black/first_frame.png as an anchor "
                   "at rope offset 0 (must match training). Skipped silently when absent.")
    p.add_argument("--use_anchor_random_count", action="store_true",
                   help="A1 arbitrary anchor count: sample k body views per case. For a "
                   "DETERMINISTIC count at inference set anchor_random_count_min == "
                   "anchor_random_count_max == k.")
    p.add_argument("--anchor_random_count_min", type=int, default=1,
                   help="Min body views with --use_anchor_random_count (default 1).")
    p.add_argument("--anchor_random_count_max", type=int, default=4,
                   help="Max body views with --use_anchor_random_count (default 4).")
    p.add_argument("--anchor_random_count_no_shuffle", action="store_true",
                   help="With --use_anchor_random_count, keep body views in CANONICAL "
                   "selected_views order (first k) instead of random.sample's "
                   "subset+permutation.")
    p.add_argument("--anchor_random_count_zero_fill", action="store_true",
                   help="With --use_anchor_random_count: keep the FULL canonical 5-slot "
                   "anchor structure — unsampled body slots become zero-fill anchors "
                   "(all-zero image at the missing view's own resolution + all-zero "
                   "person mask) at their FIXED name-mapped offsets 25/30/35/40 "
                   "(first_frame@0). Must match training.")
    p.add_argument("--anchor_random_count_priority", default=None,
                   help="Comma-separated view names fixing WHICH body views survive "
                   "no_shuffle's trim to k (e.g. \"back,front,left_side,right_side\": "
                   "k=1 keeps only back, k=2 back+front, k=3 back+front+left_side, "
                   "k=4 all four). Only meaningful with --use_anchor_random_count "
                   "--anchor_random_count_no_shuffle; names must be from "
                   "{front,left_side,right_side,back,first_frame}.")
    # --- anchor Pluecker rays ---
    p.add_argument("--use_anchor_plucker", action="store_true",
                   help="Inject per-view anchor Pluecker ray embeddings into anchor tokens, "
                   "reusing the target's trained camera_proj (shared mode). Reads "
                   "anchors_aligned_camera_info.json (w2c, aligned to the target trajectory) "
                   "from the render/anchor dirs.")
    p.add_argument("--anchor_plucker_use_camera_proj", action="store_true",
                   help="Reuse the target's trained camera_proj for the anchor rays "
                   "(must match training; the only mode implemented in this release — "
                   "no anchor_plucker_proj.* keys are expected in the checkpoint).")
    return p.parse_args()


def load_pipeline(model_root=None):
    """Load the Wan2.1-I2V-14B-720P base model via WorldCastVideoPipeline."""
    base = model_root or "pretrained/Wan2.1-I2V-14B-720P"
    model_paths = [
        ModelConfig(path=[
            f"{base}/diffusion_pytorch_model-00001-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00002-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00003-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00004-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00005-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00006-of-00007.safetensors",
            f"{base}/diffusion_pytorch_model-00007-of-00007.safetensors",
        ]),
        ModelConfig(path=f"{base}/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path=f"{base}/Wan2.1_VAE.pth"),
        ModelConfig(path=f"{base}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]
    tokenizer_config = ModelConfig(path=f"{base}/google/umt5-xxl/")

    print("Loading 14B pipeline...", flush=True)
    pipe = WorldCastVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        model_configs=model_paths,
        tokenizer_config=tokenizer_config,
    )
    return pipe


def load_checkpoint(pipe, ckpt, init_dit_from=None):
    """Load the bank checkpoint + frozen base into the pipeline.

    Splits the bank checkpoint into DiT / camera_proj / i2v_channel_adapter
    slices, warm-starts the frozen DiT (and camera_proj / i2v_channel_adapter
    when the bank checkpoint lacks them) from --init_dit_from, and returns
    (camera_proj, i2v_channel_adapter). The bank and anchor channel adapter
    are created in main() where their layer-count flags live.
    """
    dit_state = {}
    cam_state = {}
    i2v_channel_adapter_state = {}
    for k, v in ckpt.items():
        if k.startswith("camera_proj."):
            cam_state[k[len("camera_proj."):]] = v
        elif k.startswith("i2v_channel_adapter."):
            i2v_channel_adapter_state[k[len("i2v_channel_adapter."):]] = v
        else:
            dit_key = k[len("pipe.dit."):] if k.startswith("pipe.dit.") else k
            dit_state[dit_key] = v

    if init_dit_from:
        print(f"Loading Stage-1 DiT checkpoint: {init_dit_from}", flush=True)
        stage1_state = load_file(init_dit_from)
        target_state = pipe.dit.state_dict()
        stage1_dit_state = {}
        skipped = 0
        for key, value in stage1_state.items():
            dit_key = key[len("pipe.dit."):] if key.startswith("pipe.dit.") else key
            if dit_key in target_state and target_state[dit_key].shape == value.shape:
                stage1_dit_state[dit_key] = value
            elif not key.startswith(("camera_proj.", "i2v_channel_adapter.",
                                     "source_self_attn_bank.", "anchor_channel_adapter.")):
                skipped += 1
            # The frozen base release also carries the trained camera_proj and
            # i2v_channel_adapter; fill them when the bank checkpoint lacks them.
            if key.startswith("camera_proj.") and key[len("camera_proj."):] not in cam_state:
                cam_state[key[len("camera_proj."):]] = value
            elif key.startswith("i2v_channel_adapter.") and key[len("i2v_channel_adapter."):] not in i2v_channel_adapter_state:
                i2v_channel_adapter_state[key[len("i2v_channel_adapter."):]] = value
        missing_s, unexpected_s = pipe.dit.load_state_dict(stage1_dit_state, strict=False)
        print(f"  Stage-1 DiT loaded: {len(stage1_dit_state)} keys, skipped={skipped}, "
              f"missing={len(missing_s)}, unexpected={len(unexpected_s)}", flush=True)

    # Load into DiT (strict=False since bank-only checkpoints carry no DiT keys)
    missing, unexpected = pipe.dit.load_state_dict(dit_state, strict=False)
    print(f"  DiT loaded: {len(dit_state)} keys, {len(missing)} missing, {len(unexpected)} unexpected", flush=True)

    # CameraProjection: target camera conditioning + the anchor Plücker rays
    # (shared mode) both go through it — required.
    out_dim = getattr(pipe.dit, 'dim', 5120)
    if not cam_state:
        raise SystemExit(
            "ERROR: no camera_proj.* keys in --ckpt or --init_dit_from. WorldCast "
            "requires the trained camera projection module (target camera "
            "conditioning + anchor Plücker rays). Point --init_dit_from at the "
            "released frozen-base checkpoint.")
    camera_proj = CameraProjection(in_channels=6, out_dim=out_dim)
    camera_proj.load_state_dict(cam_state)
    camera_proj = camera_proj.to(device=pipe.device, dtype=pipe.torch_dtype)
    print(f"  CameraProjection loaded (out_dim={out_dim})", flush=True)

    # WanI2VChannelAdapter: required by model_fn_worldcast.
    if not i2v_channel_adapter_state:
        raise SystemExit(
            "ERROR: no i2v_channel_adapter.* keys in --ckpt or --init_dit_from. "
            "Point --init_dit_from at the released frozen-base checkpoint.")
    vae_z_dim = getattr(pipe.vae, 'z_dim', 16)
    native_i2v_channels = getattr(pipe.dit, 'in_dim', vae_z_dim + 20)
    adapter_weight = i2v_channel_adapter_state.get("proj.weight")
    if adapter_weight is not None:
        in_ch = adapter_weight.shape[1]
        out_ch = adapter_weight.shape[0]
        mid_dim = 0
    else:
        first_weight = i2v_channel_adapter_state.get("proj.0.weight")
        # Detect the LAST conv layer by max index (residual3d has proj.0,2,4,6;
        # linear-sequential has proj.0,2). proj.2 is NOT always the output layer.
        proj_idx = sorted(int(k.split(".")[1]) for k in i2v_channel_adapter_state
                          if k.startswith("proj.") and k.endswith(".weight")
                          and k.split(".")[1].isdigit())
        last_key = f"proj.{proj_idx[-1]}.weight" if proj_idx else "proj.2.weight"
        last_weight = i2v_channel_adapter_state.get(last_key)
        in_ch = first_weight.shape[1] if first_weight is not None else native_i2v_channels + vae_z_dim + 1
        out_ch = last_weight.shape[0] if last_weight is not None else native_i2v_channels
        mid_dim = first_weight.shape[0] if first_weight is not None else 0
    i2v_channel_adapter = WanI2VChannelAdapter(
        in_channels=in_ch,
        out_channels=out_ch,
        mid_channels=mid_dim,
        adapter_type="linear",
        residual_scale=1.0,
    )
    i2v_channel_adapter.load_state_dict(i2v_channel_adapter_state)
    i2v_channel_adapter = i2v_channel_adapter.to(device=pipe.device, dtype=pipe.torch_dtype)
    print(f"  I2VChannelAdapter loaded ({in_ch}ch→{out_ch}ch, mid={mid_dim})", flush=True)

    return camera_proj, i2v_channel_adapter


def load_compare_video(video_path, height, width, num_frames):
    loader = LoadVideo(
        num_frames=num_frames,
        frame_processor=lambda img: img.convert("RGB").resize((width, height), Image.BICUBIC),
    )
    return loader(video_path)


def concat_render_generated(render_path, generated_video, height, width, num_frames):
    render_video = load_compare_video(os.path.join(render_path, "render.mp4"), height, width, num_frames)
    n = min(len(render_video), len(generated_video))
    canvases = [Image.new("RGB", (width * 2, height)) for _ in range(n)]
    for i in range(n):
        gen = generated_video[i].convert("RGB").resize((width, height), Image.BICUBIC)
        canvases[i].paste(render_video[i], (0, 0))
        canvases[i].paste(gen, (width, 0))
    return canvases


def copy_conditioning(output_dir, sample_name, ref_img, render_path, anchor_path):
    """Copy reference/render/anchor conditioning next to the generated video so
    results can be eyeballed. Enabled by --copy_conditioning."""
    sdir = os.path.join(output_dir, sample_name)
    os.makedirs(sdir, exist_ok=True)
    if ref_img and os.path.isfile(ref_img):
        ext = os.path.splitext(ref_img)[1] or ".png"
        shutil.copy2(ref_img, os.path.join(sdir, "reference" + ext))
    if render_path and os.path.isdir(render_path):
        rdir = os.path.join(sdir, "render")
        os.makedirs(rdir, exist_ok=True)
        for fn in os.listdir(render_path):
            fp = os.path.join(render_path, fn)
            if os.path.isfile(fp):
                shutil.copy2(fp, os.path.join(rdir, fn))
    if anchor_path:
        if os.path.isdir(anchor_path):
            shutil.copytree(anchor_path, os.path.join(sdir, "anchors"), dirs_exist_ok=True)
        elif os.path.isfile(anchor_path):
            shutil.copy2(anchor_path, os.path.join(sdir, "anchors"))
    print(f"  Copied conditioning -> {sdir}/", flush=True)


def load_csv_data(csv_path, num_samples, shard_index=0, num_shards=1):
    """Load samples from CSV, optionally sharded across multiple workers."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if num_shards > 1 and i % num_shards != shard_index:
                continue
            rows.append(row)
            if num_samples > 0 and len(rows) >= num_samples:
                break
    return rows


def load_json_data(json_path, num_samples, shard_index=0, num_shards=1):
    """Load samples from a JSON manifest (a plain list of objects using the same
    field names as the CSV columns), with the same sharding semantics as
    load_csv_data so multigpu scripts work unchanged."""
    with open(json_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path}: expected a plain JSON list of sample objects, got {type(data).__name__}")
    required = ("prompt", "reference_image_path", "render_path")
    rows = []
    for i, row in enumerate(data):
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"{json_path}: entry {i} ({row.get('name', '?')}) missing keys: {missing}")
        if num_shards > 1 and i % num_shards != shard_index:
            continue
        rows.append(row)
        if num_samples > 0 and len(rows) >= num_samples:
            break
    return rows


def main():
    args = parse_args()
    if not args.json and not args.csv:
        raise SystemExit("ERROR: pass a manifest with --json (or --csv)")
    # Architecture validation: the released model is anchor-conditioned with a
    # trained bank; flag mismatches with the checkpoint fail at load_state_dict.
    if not args.use_viewpoint_anchors:
        raise SystemExit("ERROR: --use_viewpoint_anchors is required (WorldCast is anchor-conditioned)")
    if not args.use_source_self_attn:
        raise SystemExit("ERROR: --use_source_self_attn is required (the released checkpoints are bank-only)")
    if not args.init_dit_from:
        raise SystemExit("ERROR: --init_dit_from is required (bank-only checkpoints need the frozen base)")
    if args.use_anchor_plucker and not args.anchor_plucker_use_camera_proj:
        raise SystemExit("ERROR: --use_anchor_plucker requires --anchor_plucker_use_camera_proj "
                         "(shared target camera_proj is the only mode in this release)")
    if args.anchor_random_count_zero_fill and not args.use_anchor_random_count:
        raise SystemExit("ERROR: --anchor_random_count_zero_fill requires --use_anchor_random_count")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline and checkpoint
    pipe = load_pipeline(model_root=args.model_root)
    pipe.anchor_max_frames = args.anchor_max_frames
    pipe.use_anchor_nopad_mask = args.use_anchor_nopad_mask
    pipe.use_anchor_no_padding_black = args.use_anchor_no_padding_black
    pipe.use_anchor_random_count = args.use_anchor_random_count
    pipe.anchor_random_count_no_shuffle = args.anchor_random_count_no_shuffle
    _prio = args.anchor_random_count_priority
    if _prio:
        if isinstance(_prio, str):
            _prio = [v.strip() for v in _prio.split(",") if v.strip()]
        pipe.anchor_random_count_priority = _prio
    else:
        pipe.anchor_random_count_priority = None
    pipe.anchor_random_count_min = args.anchor_random_count_min
    pipe.anchor_random_count_max = args.anchor_random_count_max
    pipe.anchor_random_count_zero_fill = args.anchor_random_count_zero_fill
    pipe.use_first_frame_anchor = args.use_first_frame_anchor
    pipe.use_anchor_selected_views_order = args.use_anchor_selected_views_order
    pipe.use_anchor_plucker = args.use_anchor_plucker

    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    ckpt = load_file(args.ckpt)
    camera_proj, i2v_channel_adapter = load_checkpoint(
        pipe, ckpt, init_dit_from=args.init_dit_from)

    # Override model_fn (fixed architecture)
    pipe.model_fn = model_fn_worldcast
    print("Using model_fn_worldcast (anchor prefix + source self-attention bank + I2V channel adapter)", flush=True)

    # SourceSelfAttentionBank: create + load weights from ckpt
    out_dim = getattr(pipe.dit, 'dim', 5120)
    ssa_ffn = int(args.source_self_attn_inner_dim * args.source_self_attn_ffn_mult)
    source_self_attn_bank = SourceSelfAttentionBank(
        num_layers=args.source_self_attn_layers,
        interval=args.source_self_attn_interval,
        dim=out_dim,
        inner_dim=args.source_self_attn_inner_dim,
        ffn_dim=ssa_ffn,
        num_heads=args.source_self_attn_heads,
        time_embed_dim=out_dim,
        start_block=args.source_self_attn_start_block,
        use_dit_rope=args.source_self_attn_use_dit_rope,
        num_layers_is_span=not args.source_self_attn_layers_are_bank_layers,
        per_layer_proj_out=args.source_self_attn_per_layer_proj_out,
    )
    ssa_state = {k[len("source_self_attn_bank."):]: v for k, v in ckpt.items()
                 if k.startswith("source_self_attn_bank.")}
    if not ssa_state:
        raise SystemExit(f"ERROR: no source_self_attn_bank.* keys in {args.ckpt}")
    source_self_attn_bank.load_state_dict(ssa_state)
    print(f"  SourceSelfAttentionBank loaded: {len(ssa_state)} keys, layers={args.source_self_attn_layers}, "
          f"inner_dim={args.source_self_attn_inner_dim}, heads={args.source_self_attn_heads}, "
          f"per_layer_proj_out={args.source_self_attn_per_layer_proj_out}", flush=True)
    source_self_attn_bank = source_self_attn_bank.to(device=pipe.device, dtype=pipe.torch_dtype)

    # AnchorChannelAdapter: learned identity-init 1x1 conv (i2v_style 37ch ->
    # DiT in_dim 36ch) so each variable-resolution VAE anchor latent reuses
    # dit.patch_embedding.
    vae_z = getattr(pipe.vae, 'z_dim', 16)
    in_dim = getattr(pipe.dit, 'in_dim', vae_z + 20)
    if 2 * vae_z + 4 != in_dim:
        raise SystemExit(f"ERROR: i2v_style anchor packing needs 2*{vae_z}+4 == in_dim {in_dim}")
    adapter_in = 2 * vae_z + 4 + 1
    anchor_channel_adapter = AnchorChannelAdapter(
        in_channels=adapter_in, out_dim=in_dim, init="identity")
    ada_state = {k[len("anchor_channel_adapter."):]: v for k, v in ckpt.items()
                 if k.startswith("anchor_channel_adapter.")}
    if not ada_state:
        raise SystemExit(f"ERROR: no anchor_channel_adapter.* keys in {args.ckpt}")
    anchor_channel_adapter.load_state_dict(ada_state)
    print(f"  AnchorChannelAdapter loaded: {len(ada_state)} keys ({adapter_in}ch→{in_dim}ch, "
          f"conv mode, i2v_style)", flush=True)
    anchor_channel_adapter = anchor_channel_adapter.to(device=pipe.device, dtype=pipe.torch_dtype)

    # Anchor Plücker shared mode: anchor rays go through the target's
    # camera_proj (loaded above) — no anchor_plucker_proj module exists.
    anchor_plucker_use_camera_proj = args.use_anchor_plucker and args.anchor_plucker_use_camera_proj

    # Load samples from manifest
    if args.json:
        samples = load_json_data(args.json, args.num_samples, shard_index=args.shard_index, num_shards=args.num_shards)
    else:
        samples = load_csv_data(args.csv, args.num_samples, shard_index=args.shard_index, num_shards=args.num_shards)
    print(f"Loaded {len(samples)} samples (shard {args.shard_index}/{args.num_shards})", flush=True)

    # Trajectory processing (--traj): replace each row with one row per
    # requested trajectory, building the conditioning dirs on the way
    # (scripts/data/trajectories.py; resumable — complete dirs are skipped).
    if args.traj:
        from scripts.data.trajectories import expand_samples
        traj_names = [t.strip() for t in args.traj.split(",") if t.strip()]
        samples = expand_samples(
            samples, traj_names, out_root=args.traj_out_root,
            nframes=args.traj_nframes or args.num_frames)
        print(f"Trajectory expansion: {len(traj_names)} preset(s) per case -> "
              f"{len(samples)} generation rows", flush=True)

    print(f"\nGenerating {len(samples)} samples (steps={args.num_inference_steps}, "
          f"cfg={args.cfg_scale}, sigma_shift={args.sigma_shift}, seed={args.seed})\n", flush=True)

    for i, row in enumerate(samples):
        prompt = row["prompt"]
        ref_img = row["reference_image_path"]
        render_path = row.get("render_path", "")

        # Use the parent directory as the ID when datasets use a generic video filename.
        video_path = row.get("video", "")
        if video_path:
            video_path = os.path.normpath(video_path)
            video_stem = os.path.splitext(os.path.basename(video_path))[0]
            if video_stem in {"gt_video", "gt_81f_16fps", "video", "gt_uniform_81_h264", "gt_uniform_81"}:
                sample_name = os.path.basename(os.path.dirname(video_path))
            else:
                sample_name = video_stem
        else:
            sample_name = os.path.basename(os.path.normpath(render_path))

        print(f"[{i+1}/{len(samples)}] {sample_name}", flush=True)

        # Skip if output already exists (resumable inference)
        out_path = os.path.join(args.output_dir, f"{sample_name}_seed{args.seed}{args.output_suffix}.mp4")
        if os.path.exists(out_path):
            print(f"  SKIP: already exists ({out_path})", flush=True)
            continue
        print(f"  prompt: {prompt[:80]}", flush=True)

        input_image = Image.open(ref_img).convert("RGB")

        video = pipe(
            prompt=prompt,
            reference_image_path=ref_img,
            render_path=render_path,
            input_image=input_image,
            seed=args.seed,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            nframe=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
            camera_proj=camera_proj,
            i2v_channel_adapter=i2v_channel_adapter,
            source_self_attn_bank=source_self_attn_bank,
            anchor_channel_adapter=anchor_channel_adapter,
            invert_camera_extrinsic=args.invert_camera_extrinsic,
            **({
                "viewpoint_anchors_path": row["viewpoint_anchors_path"],
                "viewpoint_anchor_frame_offset": args.viewpoint_anchor_frame_offset,
            } if args.use_viewpoint_anchors and row.get("viewpoint_anchors_path") else {}),
            anchor_plucker_use_camera_proj=anchor_plucker_use_camera_proj,
        )

        save_video(video, out_path, fps=15, quality=5)
        print(f"  Saved: {out_path}", flush=True)
        if args.save_render_concat:
            concat_video = concat_render_generated(render_path, video, args.height, args.width, args.num_frames)
            concat_path = os.path.splitext(out_path)[0] + "_render_gen.mp4"
            save_video(concat_video, concat_path, fps=15, quality=5)
            print(f"  Saved render/gen concat: {concat_path}", flush=True)
        if args.copy_conditioning:
            copy_conditioning(args.output_dir, sample_name, ref_img, render_path, row.get("viewpoint_anchors_path", ""))
        print("", flush=True)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
