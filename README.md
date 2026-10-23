<div align="center">

<img src="./assets/logo.png" width="42%"/>

# WorldCast: World-Consistent Character Video Generation with Large-Angle Camera Control

<a href="#"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv%20%C2%B7%20coming%20soon&color=red"></a>
&ensp;
<a href="https://pixctrol.github.io/WorldCast/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Page&color=green"></a>
&ensp;
<a href="https://github.com/PixCtrol/WorldCast"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue"></a>
&ensp;
<a href="#"><img src="https://img.shields.io/static/v1?label=Checkpoints&message=coming%20soon&color=yellow"></a>
&ensp;
<a href="#"><img src="https://img.shields.io/static/v1?label=CLACT%20Dataset&message=coming%20soon&color=orange"></a>

**Lei Cao**<sup>1,2\*</sup> · **Yu Liu**<sup>2\*</sup> · **Wenhua Liao**<sup>2</sup> · **Peng Zhang**<sup>2‡</sup> · **Fei Wang**<sup>2</sup> · **Hao Zhang**<sup>1</sup> · **Jiajun Le**<sup>1</sup> · **Daiguo Zhou**<sup>2†</sup> · **Jian Luan**<sup>2</sup> · **Jiayi Ma**<sup>1,3†</sup>

<sup>1</sup>Electronic Information School, Wuhan University &ensp; <sup>2</sup>MiLM Plus, Xiaomi Inc. &ensp; <sup>3</sup>School of Robotics, Wuhan University

<sup>\*</sup>Equal Contribution &ensp; <sup>‡</sup>Project Lead &ensp; <sup>†</sup>Co-corresponding Authors

</div>

Given one reference image, a target camera trajectory, and **1–4 viewpoint
anchor images** of the subject (from arbitrary viewpoints), WorldCast
generates a video that follows the target camera motion while preserving the
subject's identity across viewpoints.

## News

- **[2026.10]** Initial release: inference code, bundled demo case, and project page.

## TODO

- [x] Release inference code and demo
- [ ] Release paper (arXiv)
- [ ] Release model checkpoints
- [ ] Release CLACT dataset
- [ ] Release training code

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/PixCtrol/WorldCast
cd WorldCast
```

### 2. Set Up the Environment

Python >= 3.10, a CUDA GPU (>= 48 GB recommended for the 14B model at
480x832; memory scales with resolution and frame count), bf16.

```bash
pip install -r requirements.txt
```

### 3. Download Weights

| Component | Source | Place at |
|---|---|---|
| Wan2.1-I2V-14B-720P base | HuggingFace [`Wan-AI/Wan2.1-I2V-14B-720P`](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) | `pretrained/Wan2.1-I2V-14B-720P/` |
| WorldCast checkpoints (frozen base snapshot + trained bank + anchor adapter) | coming soon | `checkpoints/` |

### 4. Run Demos

Bundled five-trajectory demo (preprocessed case included):

```bash
python demo.py \
  --manifest demo_case_traj5.json \
  --model_root pretrained/Wan2.1-I2V-14B-720P \
  --ckpt checkpoints/worldcast_bank.safetensors \
  --init_dit_from checkpoints/base_dit.safetensors
```

### Few-Step Sampling

By default sampling runs 50 steps / cfg 5.0. Optionally fuse a
step-distillation LoRA (lightx2v / FusionX style, **not** part of this
release — bring your own) and sample in 6 steps at cfg 1.0:

```bash
python demo.py \
  --manifest demo_case_traj5.json \
  --model_root pretrained/Wan2.1-I2V-14B-720P \
  --ckpt checkpoints/worldcast_bank.safetensors \
  --init_dit_from checkpoints/base_dit.safetensors \
  --accel_lora_path <lightx2v_fusionx_i2v14b.safetensors> \
  --num_inference_steps 6 --cfg_scale 1.0
```

## Trajectories

`--traj` presets (built by `scripts/data/trajectories.py`):

| Trajectory | Motion |
|---|---|
| `flatring` | horizontal 360° ring about the subject |
| `lemniscate` | Gerono figure-8 in the first-frame screen plane |
| `smooth_flatring_tilt20` | 360° ring pivoted to a 20° tilt |
| `180_push` | 180° arc with a sinusoidal push-pull radius ripple |
| `zoom_out` | one-way dolly out from the subject |

Parameterized families are also accepted: `smooth_flatring_tilt<N>` (tilt
1–89°) and `<N>_push` (sweep 30–360°).

For batch inference over custom cases, see `scripts/infer/infer_worldcast.py`
and `launch/infer/run_anchor_inference_example.sh`; the manifest format is
shown in `examples/anchor_case_manifest.example.json`, and the bundled
`demo_case/` illustrates the full case layout.

## Citation

If WorldCast is useful for your research, please cite our paper. BibTeX will
be updated after publication.

```bibtex
@misc{worldcast,
  title   = {WorldCast: World-Consistent Character Video Generation with Large-Angle Camera Control},
  author  = {Cao, Lei and Liu, Yu and Liao, Wenhua and Zhang, Peng and Wang, Fei and Zhang, Hao and Le, Jiajun and Zhou, Daiguo and Luan, Jian and Ma, Jiayi},
  year    = {2026}
}
```

## License

Apache-2.0 (inherited from DiffSynth-Studio). Wan base weights follow their
own license from Wan-AI.

## Acknowledgements

WorldCast is built on
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

### Dataset acknowledgment

This work uses **SpatialVID** under **CC BY-NC-SA 4.0** and its Terms of Use.
The authors confirm that all uses of this dataset are strictly for academic
research purposes and not for any commercial applications.
