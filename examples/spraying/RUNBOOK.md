# MolmoAct v1 — Spraying fine-tune runbook

End-to-end recipe to fine-tune **MolmoAct-7B-D-0812** on the spraying dataset
(LoRA, depth-tokens enabled). The same 9-dim/2-camera/224×224 LeRobot dataset
used for the openpi-10x pi0.5 run is the input. We picked v1 because the
official MolmoAct **v2** training code isn't released yet — `allenai/molmoact2`
ships only README + an inference-only LeRobot policy at the time of writing.

> Hardware: 8× A100/H100 recommended for full training. Single-GPU with smaller
> `--global_batch_size` works for smoke tests.

## Schema (unchanged from openpi-10x)

| dim | source                       | name           | units | trained as |
|----:|------------------------------|----------------|-------|------------|
| 0   | `observation.state[0]`       | liftkit_mid    | m     | delta      |
| 1   | `observation.state[4]`       | liftkit_top    | m     | delta      |
| 2   | `observation.state[8]`       | shoulder_pan   | rad   | delta      |
| 3   | `observation.state[9]`       | shoulder_lift  | rad   | delta      |
| 4   | `observation.state[10]`      | elbow          | rad   | delta      |
| 5   | `observation.state[11]`      | wrist_1        | rad   | delta      |
| 6   | `observation.state[12]`      | wrist_2        | rad   | delta      |
| 7   | `observation.state[13]`      | wrist_3        | rad   | delta      |
| 8   | `observation.tool_state[3]`  | sprayer_pwr    | bool  | absolute   |

Cameras: `image` (base) + `wrist_image` (wrist), both 224×224. The
preprocessor's `Point` head only consults the base image; the wrist camera is
attached as a second view at training time via `--max_images 2`.

## Step 0 — Source dataset

This pipeline consumes the **already-converted** LeRobot dataset that
openpi-10x produced. Re-run the openpi conversion if it isn't already on the VM:

```bash
cd ~/Documents/git/openpi-10x
unset PYTHONPATH
PYTHONPATH=. uv run python examples/spraying/convert_spraying_data_to_lerobot.py \
    --src_root /path/to/source/dataset \
    --repo_id rishi-10x/spraying-v1-local \
    --overwrite
```

Output lands in `$HF_LEROBOT_HOME/rishi-10x/spraying-v1-local`
(default `~/.cache/huggingface/lerobot/rishi-10x/spraying-v1-local`).

## Step 1 — Install MolmoAct + Depth-Anything-V2 on the VM

```bash
cd ~/Documents/git
git clone https://github.com/allenai/molmoact.git
git clone https://github.com/DepthAnything/Depth-Anything-V2.git

# MolmoAct env (Python 3.11 + your CUDA-matched PyTorch first)
cd molmoact
pip install -e .[all]

# Data-preprocessing extras
cd ../Depth-Anything-V2
pip install -r requirements.txt
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-python-headless --no-cache-dir
pip install lerobot==0.3.3
```

## Step 2 — Fetch checkpoints

```bash
mkdir -p ~/Documents/git/Depth-Anything-V2/checkpoints

# Depth-Anything-V2 backbone (~390 MB, vitb)
wget -O ~/Documents/git/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth \
    https://huggingface.co/allenai/MolmoAct-7B-D-0812/resolve/main/depth_anything_v2_vitb.pth

# MolmoAct depth-token VQVAE
wget -O ~/Documents/git/molmoact/vae-final.pt \
    https://huggingface.co/allenai/MolmoAct-7B-D-0812/resolve/main/vae-final.pt
```

The MolmoAct base checkpoint (`allenai/MolmoAct-7B-D-0812`) is pulled by the
training script via `huggingface_hub` on first use; pre-warm if your VM has
limited egress:

```bash
huggingface-cli download allenai/MolmoAct-7B-D-0812
```

## Step 3 — Preprocess into Action-Reasoning Data

This adds `depth`, `trace`, and `processed_action` to every frame. Output is a
HuggingFace `datasets.save_to_disk` directory plus a `dataset_statistics.json`.

**Spraying-specific overrides** (added in patches to upstream — see "Patches" at
the bottom of this file):

- `--point-prompt "point to the spray nozzle"` — Molmo's gripper prompt finds
  nothing on a sprayer end-effector; this is the spraying-specific prompt.
- `--normalize-dims 8` — normalize all 8 joint dims; leave dim 8 (binary
  sprayer trigger) unnormalized. Default is 6, which would wrongly leave
  `wrist_2` and `wrist_3` un-normalized.

```bash
cd ~/Documents/git/molmoact

export DEPTH_CHECKPOINT_DIR=~/Documents/git/Depth-Anything-V2/checkpoints
export VQVAE_MODEL_PATH=~/Documents/git/molmoact/vae-final.pt
export PYTHONPATH=$PYTHONPATH:~/Documents/git/Depth-Anything-V2:~/Documents/git/molmoact

python preprocess/action_reasoning_data.py \
    --dataset-path rishi-10x/spraying-v1-local \
    --output-path ~/data/molmoact/spraying-v1-processed \
    --depth-encoder vitb \
    --line-length 5 \
    --process-actions \
    --action-bins 256 \
    --action-chunk-size 8 \
    --normalize-dims 8 \
    --point-prompt "point to the spray nozzle"
```

`--dataset-path` is a LeRobot v3 repo_id resolved against `$HF_LEROBOT_HOME`;
the existing converter writes to `rishi-10x/spraying-v1-local` by default.

After this completes, verify:

```bash
ls ~/data/molmoact/spraying-v1-processed
# expect: data files + dataset_info.json + state.json + dataset_statistics.json

python -c "
from datasets import load_from_disk
ds = load_from_disk('$HOME/data/molmoact/spraying-v1-processed')
print('frames:', len(ds))
print('keys:', list(ds.features.keys()))
sample = ds[0]
print('depth tokens (head):', sample['depth'][:80])
print('trace:', sample['trace'])
print('language_instruction:', sample['language_instruction'])
print('processed_action keys:', list(__import__('json').loads(sample['processed_action']).keys()))
"
```

> Quick sanity check on the gripper-point prompt: scan a few frames and
> confirm `trace` is non-empty for the bulk of episodes. If most frames have
> `trace == "[]"`, the Molmo prompt isn't catching the end-effector — try
> `"point to the end of the spray gun"` or `"point to the orange spray
> nozzle"` (use whatever color/feature is most distinctive in your data).

## Step 4 — Smoke-test the data loader

```bash
cd ~/Documents/git/molmoact
python -c "
from olmo.data.custom_lerobot_dataset import CustomLeRobotDataset
import numpy as np
ds = CustomLeRobotDataset(
    path='$HOME/data/molmoact/spraying-v1-processed',
    high_res=False, style='demo')
sample = ds.get(0, np.random.RandomState(0))
print('image count:', len(sample['image']))
print('image[0] size:', sample['image'][0].size)
print('answer head:', sample['answers'][:200])
"
```

Expect 2 images and an answer string containing `<DEPTH_START>...<DEPTH_END>`
plus a chunked-action token sequence.

## Step 5 — Launch training

The `robot-finetune` mixture in `launch_scripts/train_multitask_model.py` now
reads `MOLMOACT_FINETUNE_PATH` so you don't have to edit the file per-run.

```bash
cd ~/Documents/git/molmoact

export MOLMOACT_FINETUNE_PATH=~/data/molmoact/spraying-v1-processed
export RANK=0 ADDR=127.0.0.1 PORT=29500

WANDB_API_KEY=<your_wandb_api_key> torchrun \
    --nnodes=1 --nproc-per-node=8 \
    --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    launch_scripts/train_multitask_model.py \
    robot-finetune allenai/MolmoAct-7B-D-0812 \
    --wandb.name=molmoact_spraying_lora_run1 \
    --wandb.entity=<entity> \
    --wandb.project=vla-v0 \
    --norm_stats_path ~/data/molmoact/spraying-v1-processed/dataset_statistics.json \
    --save_folder=checkpoints/molmoact_spraying_lora \
    --save_overwrite \
    --duration 10000 \
    --ft_embedding all \
    --depth_tokens \
    --global_batch_size 16 \
    --lr_connector 5e-4 \
    --lr_vit 5e-4 \
    --lr_llm 5e-4 \
    --save_interval 2000 \
    --save_num_checkpoints_to_keep 5 \
    --max_images 2 \
    --lora_enable --lora_rank 32 --lora_alpha 16 --lora_dropout 0.0 \
    --img_aug
```

- `--max_images 2` enables the wrist camera as a second view.
- `--depth_tokens` is required because preprocessing produced depth tokens.
- `--norm_stats_path` must point at the `dataset_statistics.json` written in
  Step 3, not at any older copy.
- For < 8 GPUs, drop `--global_batch_size` proportionally and bump
  `--duration` to keep step-count comparable.

## Step 6 — Merge LoRA + convert to HF for inference

After training, the checkpointer writes sharded base + a `stepXXX-lora`
adapter. Merge, then HF-convert:

```bash
STEP=10000
RUN=checkpoints/molmoact_spraying_lora

python3 -m scripts.merge_lora \
    --base_dir $(huggingface-cli download allenai/MolmoAct-7B-D-0812 --local-dir-use-symlinks False) \
    --lora_dir ${RUN}/step${STEP}-lora \
    --output_dir ${RUN}/step${STEP}-merge

python3 -m olmo.hf_model.molmoact.convert_molmoact_to_hf \
    --checkpoint_dir ${RUN}/step${STEP}-merge \
    --output_dir ${RUN}/step${STEP}-hf \
    --style demo \
    --norm_stats_path ~/data/molmoact/spraying-v1-processed/dataset_statistics.json

# Smoke-test inference (2-camera frame from any episode)
python3 olmo/hf_model/molmoact/test_molmoact.py \
    --checkpoint_dir ${RUN}/step${STEP}-hf \
    --images /path/to/base_frame.png /path/to/wrist_frame.png \
    --instruction "spray the surface" \
    --unnorm_key spraying-v1-processed
```

`--unnorm_key` matches the dataset key recorded inside
`dataset_statistics.json` (it's the basename of the processed-dataset path).

## Patches applied to this clone

These are local edits to upstream MolmoAct, made so the spraying schema fits
without forking. Keep them in mind if you `git pull` upstream:

1. `preprocess/processors.py`
   - `Point.__init__(prompt=...)` — make Molmo's end-effector prompt configurable.
   - `Point.inference_point` — passes `self.prompt` through to `point_at_gripper`.
   - `ActionProcessor.__init__(normalize_dims=...)` — store normalize-dims on the instance.
   - `ActionProcessor.compute_dataset_statistics` — defaults to `self.normalize_dims` when not overridden.

2. `preprocess/action_reasoning_data.py`
   - New CLI flags `--point-prompt` and `--normalize-dims`, threaded through `DatasetProcessor` → `Point` / `ActionProcessor`.

3. `launch_scripts/train_multitask_model.py`
   - `robot-finetune` mixture now reads `MOLMOACT_FINETUNE_PATH` env var, falling back to the upstream `/path/to/processed_dataset` placeholder.

## Troubleshooting

- **`ModuleNotFoundError: depth_anything_v2`** → `PYTHONPATH` must include the
  Depth-Anything-V2 clone (the preprocessor `sys.path.append`s its sibling, so
  the two repos must live next to each other under a common parent, or you
  must export PYTHONPATH explicitly as in Step 3).
- **Most frames have `trace == "[]"`** → end-effector prompt isn't matching;
  iterate on `--point-prompt`.
- **`norm_stats.json` mask all True / all False** → wrong `--normalize-dims`;
  for spraying use 8.
- **OOM during training** → drop `--global_batch_size` and/or set
  `--device_train_microbatch_size 1`.
- **Diverging loss with binary tool dim** → confirm `mask[8] == false` in
  `dataset_statistics.json`; the binary trigger should NOT be normalized.

## Hardware sanity (vs openpi pi0.5)

| Model            | Trainable params (LoRA) | Min VRAM | Notes                         |
|------------------|-------------------------|---------:|-------------------------------|
| pi0.5 (LoRA)     | ~467 M                  | 22.5 GB | 4090 OK                       |
| MolmoAct-7B-D    | ~1+ G with LoRA32       | ≥ 1×A100 80 GB or 8× A100 40 GB | 4090 will be tight; expect to drop batch/microbatch |
