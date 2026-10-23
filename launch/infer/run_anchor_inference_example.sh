#!/usr/bin/env bash
# run_anchor_inference_example.sh — anchor-conditioned video generation,
# single GPU, single case (or small manifest).
#
# This template mirrors the flag set of the ANCHOR-DROP (zero-fill-slot)
# production line: the first frame enters at rope offset 0 and the four
# body-view slots (front / left_side / right_side / back) at 25/30/35/40,
# with anchor-drop enabled — dropped slots become zero images at their
# natural resolution with zero masks, and rope offsets stay fixed per view
# name. KCOUNT=4 keeps all four body views in canonical order (the
# checkpoint's standard eval config); lower KCOUNT to actually drop slots
# (e.g. KCOUNT=3 drops one view, chosen by ANCHOR_PRIORITY).
#
# Expected inputs (see README.md "Data format"):
#   $MODEL_ROOT    Wan2.1-I2V-14B-720P base weights (HF: Wan-AI/Wan2.1-I2V-14B-720P)
#   $CKPT          trained source-self-attention bank checkpoint (bank + anchor_channel_adapter)
#   $INIT_DIT_FROM frozen base DiT checkpoint the bank was trained against
#                  (provides frozen DiT + camera_proj + i2v channel adapter)
#   $JSON          case manifest (see examples/anchor_case_manifest.example.json)
#
# Usage (repo root):
#   bash launch/infer/run_anchor_inference_example.sh
#
# Env overrides: PYTHON MODEL_ROOT CKPT INIT_DIT_FROM JSON KCOUNT
#   ANCHOR_PRIORITY HEIGHT WIDTH NUM_FRAMES NUM_INFER_STEPS CFG_SCALE
#   SIGMA_SHIFT SEED GPUS OUTPUT_DIR
set -u -o pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

MODEL_ROOT="${MODEL_ROOT:-pretrained/Wan2.1-I2V-14B-720P}"
CKPT="${CKPT:-checkpoints/worldcast_bank.safetensors}"
INIT_DIT_FROM="${INIT_DIT_FROM:-checkpoints/base_dit.safetensors}"
JSON="${JSON:-examples/anchor_case_manifest.example.json}"

# anchor-drop (zero-fill-slot) settings: KCOUNT body views kept, canonical
# slot order, slots dropped from the END of the priority list
KCOUNT="${KCOUNT:-4}"
ANCHOR_PRIORITY="${ANCHOR_PRIORITY:-back,front,left_side,right_side}"

HEIGHT="${HEIGHT:-832}"
WIDTH="${WIDTH:-480}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFER_STEPS="${NUM_INFER_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-5.0}"
SIGMA_SHIFT="${SIGMA_SHIFT:-2.0}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

for f in "$MODEL_ROOT/diffusion_pytorch_model-00001-of-00007.safetensors" \
         "$CKPT" "$INIT_DIT_FROM" "$JSON"; do
  [ -f "$f" ] || { echo "[ERROR] missing input: $f" >&2; exit 1; }
done

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/infer_gpu${GPUS%% *}.log"

echo "[anchor-infer] gpu=$GPUS H=$HEIGHT W=$WIDTH steps=$NUM_INFER_STEPS cfg=$CFG_SCALE seed=$SEED"
echo "[anchor-infer] anchor-drop k=$KCOUNT priority=$ANCHOR_PRIORITY"
echo "[anchor-infer] model_root=$MODEL_ROOT"
echo "[anchor-infer] ckpt=$CKPT init_dit=$INIT_DIT_FROM"
echo "[anchor-infer] json=$JSON -> $OUTPUT_DIR (log: $LOG)"

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -u scripts/infer/infer_worldcast.py \
  --model_root "$MODEL_ROOT" \
  --ckpt "$CKPT" --init_dit_from "$INIT_DIT_FROM" \
  --json "$JSON" \
  --use_viewpoint_anchors \
  --anchor_max_frames 4 \
  --use_anchor_nopad_mask --use_anchor_no_padding_black \
  --use_anchor_selected_views_order --use_first_frame_anchor \
  --use_anchor_random_count \
  --anchor_random_count_min "$KCOUNT" --anchor_random_count_max "$KCOUNT" \
  --anchor_random_count_zero_fill --anchor_random_count_no_shuffle \
  --anchor_random_count_priority "$ANCHOR_PRIORITY" \
  --use_anchor_plucker --anchor_plucker_use_camera_proj \
  --use_source_self_attn --source_self_attn_layers 20 \
  --source_self_attn_layers_are_bank_layers \
  --source_self_attn_per_layer_proj_out --source_self_attn_interval 2 \
  --source_self_attn_start_block 0 --source_self_attn_heads 8 \
  --source_self_attn_inner_dim 1024 --source_self_attn_use_dit_rope \
  --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" \
  --num_samples 1000 --shard_index 0 --num_shards 1 \
  --num_inference_steps "$NUM_INFER_STEPS" --cfg_scale "$CFG_SCALE" \
  --sigma_shift "$SIGMA_SHIFT" --seed "$SEED" --save_render_concat \
  --output_dir "$OUTPUT_DIR" > "$LOG" 2>&1
RC=$?

echo ""
echo "[DONE] rc=$RC"
ls -la "$OUTPUT_DIR" 2>/dev/null
exit "$RC"
