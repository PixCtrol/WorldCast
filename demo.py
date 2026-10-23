#!/usr/bin/env python3
"""WorldCast demo — one subject, five camera trajectories.

Generates one 81-frame video per trajectory (flat 360-deg ring / Gerono
lemniscate / 20-deg tilted flat ring / 180-deg push-pull arc / zoom-out) for
the bundled demo case (a curly-haired boy in a science museum,
`demo_case/A360-031__portrait`) using the released anchor-conditioned
checkpoint:

    python demo.py \
        --manifest demo_case_traj5.json \
        --model_root pretrained/Wan2.1-I2V-14B-720P \
        --ckpt checkpoints/worldcast_bank.safetensors \
        --init_dit_from checkpoints/base_dit.safetensors

`--manifest` is a plain JSON list of case rows (same schema as
examples/anchor_case_manifest.example.json), one row per trajectory with a
distinct `case` id. The bundled `demo_case_traj5.json` references the five
prebuilt trajectory conditioning dirs under `demo_case/`; they were
generated from the bundled base case dir (`demo_case/A360-031__portrait`)
by `scripts/data/trajectories.py`, which can rebuild them or produce other
trajectories (see README "Trajectory generation").

The architecture below is FIXED to the released checkpoint's training config
(anchor-drop line: first-frame anchor at rope offset 0 + four body-view slots at
25/30/35/40 with zero-fill dropping, per-anchor Pluecker rays through the
shared target camera_proj, 20-layer source self-attention bank reusing the
DiT RoPE). Only sampling parameters are exposed as flags — changing the
architecture would fail at load_state_dict anyway.

The 14B pipeline is loaded ONCE, then each manifest row is generated in turn.
Existing outputs are skipped, so an interrupted demo resumes where it stopped.
"""
import argparse
import json
import os
import sys

from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "scripts", "infer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from safetensors.torch import load_file

from infer_worldcast import load_pipeline, load_checkpoint, concat_render_generated
from diffsynth.pipelines.worldcast_pipeline_interleave import (
    model_fn_worldcast, SourceSelfAttentionBank, AnchorChannelAdapter,
)
from diffsynth.utils.data import save_video


# ---- fixed architecture: the released anchor-drop checkpoint's training config ----
BANK_CONFIG = dict(
    num_layers=20, interval=2, inner_dim=1024, ffn_dim=4096, num_heads=8,
    start_block=0, use_dit_rope=True,
    num_layers_is_span=False,        # --source_self_attn_layers_are_bank_layers
    per_layer_proj_out=True,         # ffn_dim = inner_dim * ffn_mult 4.0
)
ANCHOR_PRIORITY = ["back", "front", "left_side", "right_side"]
ANCHOR_K = 4                         # body views kept (4 = all, standard eval)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True,
                   help="demo manifest json (plain list of case rows)")
    p.add_argument("--model_root", default="pretrained/Wan2.1-I2V-14B-720P",
                   help="Wan2.1-I2V-14B-720P base weights root")
    p.add_argument("--ckpt", default="checkpoints/worldcast_bank.safetensors",
                   help="trained bank checkpoint (bank + anchor_channel_adapter)")
    p.add_argument("--init_dit_from", default="checkpoints/base_dit.safetensors",
                   help="frozen base DiT checkpoint the bank was trained against")
    p.add_argument("--output_dir", default="outputs_demo")
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--sigma_shift", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_render_concat", action="store_true",
                   help="skip the side-by-side render|generated comparison videos")
    p.add_argument("--invert_camera_extrinsic", action="store_true",
                   help="invert cam_info extrinsic (w2c->c2w); only for old "
                        "w2c data — demo manifests are c2w, leave off")
    return p.parse_args()


def build_bank(pipe, ckpt):
    out_dim = getattr(pipe.dit, "dim", 5120)
    bank = SourceSelfAttentionBank(dim=out_dim, time_embed_dim=out_dim,
                                   **BANK_CONFIG)
    state = {k[len("source_self_attn_bank."):]: v for k, v in ckpt.items()
             if k.startswith("source_self_attn_bank.")}
    if not state:
        raise SystemExit("ERROR: no source_self_attn_bank.* keys in the checkpoint")
    bank.load_state_dict(state)
    bank = bank.to(device=pipe.device, dtype=pipe.torch_dtype)
    print(f"  SourceSelfAttentionBank: {len(state)} keys "
          f"({BANK_CONFIG['num_layers']} layers, per-layer proj_out, DiT RoPE)",
          flush=True)
    return bank


def build_anchor_adapter(pipe, ckpt):
    vae_z = getattr(pipe.vae, "z_dim", 16)
    in_dim = getattr(pipe.dit, "in_dim", vae_z + 20)
    if 2 * vae_z + 4 != in_dim:
        raise SystemExit(f"ERROR: i2v_style anchor packing needs "
                         f"2*{vae_z}+4 == in_dim {in_dim}")
    adapter = AnchorChannelAdapter(in_channels=2 * vae_z + 4 + 1,
                                   out_dim=in_dim, init="identity")
    state = {k[len("anchor_channel_adapter."):]: v for k, v in ckpt.items()
             if k.startswith("anchor_channel_adapter.")}
    if not state:
        raise SystemExit("ERROR: no anchor_channel_adapter.* keys in the checkpoint")
    adapter.load_state_dict(state)
    adapter = adapter.to(device=pipe.device, dtype=pipe.torch_dtype)
    print(f"  AnchorChannelAdapter: {len(state)} keys "
          f"({2 * vae_z + 5}ch -> {in_dim}ch, i2v_style)", flush=True)
    return adapter


def main():
    args = parse_args()
    with open(args.manifest) as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise SystemExit(f"{args.manifest}: expected a plain JSON list")
    for i, row in enumerate(samples):
        for k in ("case", "prompt", "reference_image_path", "render_path",
                  "viewpoint_anchors_path"):
            if k not in row:
                raise SystemExit(f"{args.manifest}: row {i} missing '{k}'")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- pipeline + fixed anchor-drop wiring (loads once) ----
    print(f"Loading pipeline from {args.model_root}", flush=True)
    pipe = load_pipeline(model_root=args.model_root)
    pipe.anchor_max_frames = 4
    pipe.use_anchor_nopad_mask = True
    pipe.use_anchor_no_padding_black = True
    pipe.use_anchor_random_count = True
    pipe.anchor_random_count_no_shuffle = True
    pipe.anchor_random_count_priority = ANCHOR_PRIORITY
    pipe.anchor_random_count_min = ANCHOR_K
    pipe.anchor_random_count_max = ANCHOR_K
    pipe.anchor_random_count_zero_fill = True
    pipe.use_first_frame_anchor = True
    pipe.use_anchor_selected_views_order = True
    pipe.use_anchor_plucker = True
    pipe.model_fn = model_fn_worldcast

    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    ckpt = load_file(args.ckpt)
    camera_proj, i2v_channel_adapter = load_checkpoint(
        pipe, ckpt, init_dit_from=args.init_dit_from)
    source_self_attn_bank = build_bank(pipe, ckpt)
    anchor_channel_adapter = build_anchor_adapter(pipe, ckpt)

    print(f"\nGenerating {len(samples)} trajectories (steps="
          f"{args.num_inference_steps}, cfg={args.cfg_scale}, seed={args.seed})\n",
          flush=True)

    # ---- one video per trajectory ----
    for i, row in enumerate(samples):
        case_id = row["case"]
        print(f"[{i + 1}/{len(samples)}] {case_id}  (trajectory: "
              f"{row.get('trajectory', '?')})", flush=True)
        out_path = os.path.join(args.output_dir,
                                f"{case_id}_seed{args.seed}.mp4")
        if os.path.exists(out_path):
            print(f"  SKIP: already exists ({out_path})", flush=True)
            continue
        print(f"  prompt: {row['prompt'][:80]}", flush=True)

        ref_img = row["reference_image_path"]
        input_image = Image.open(ref_img).convert("RGB")

        video = pipe(
            prompt=row["prompt"],
            reference_image_path=ref_img,
            render_path=row["render_path"],
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
            viewpoint_anchors_path=row["viewpoint_anchors_path"],
            # (no viewpoint_anchor_frame_offset: the anchor-drop line maps each anchor
            # NAME to a fixed rope slot 0/25/30/35/40 via viewpoint_anchor_traj_offsets,
            # which replaces the ordinal frame_offset path in _build_anchor_rope)
            anchor_plucker_use_camera_proj=True,
        )

        save_video(video, out_path, fps=15, quality=5)
        print(f"  Saved: {out_path}", flush=True)
        if not args.no_render_concat:
            concat = concat_render_generated(row["render_path"], video,
                                             args.height, args.width,
                                             args.num_frames)
            concat_path = os.path.splitext(out_path)[0] + "_render_gen.mp4"
            save_video(concat, concat_path, fps=15, quality=5)
            print(f"  Saved render/gen concat: {concat_path}", flush=True)
        print("", flush=True)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
