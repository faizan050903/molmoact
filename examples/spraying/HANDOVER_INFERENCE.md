# MolmoAct Spraying — Inference Smoke Test Handover

This document hands off the in-progress inference smoke test for our
MolmoAct-7B-D LoRA finetune on the spraying dataset. Training is done.
What's left: prove the trained adapter works end-to-end (predicted
action vs. ground truth) on a local 2×16 GB box.

---

## Prompt for the new Claude Code session

Paste this verbatim into a fresh Claude Code session running on the
target machine (`cd` to wherever you clone the repo first):

> I'm continuing work on the MolmoAct v1 LoRA finetune for our 9-dim spraying robot. Training is done (20k steps, run `molmoact_spraying_cleaned_lora_v1`, commit `9df0ad3`). The inference smoke test in `examples/spraying/inference.py` runs, but the LoRA contribution to the forward pass is currently zero — outputs are bit-identical with and without `model.disable_adapter()`. We've been iterating on the key-rewrite from training-time FSDP-wrapped names to HF-PEFT names. The latest commit on `spraying-finetune` (`75b9b19`) adds a probe at `[3a/5]` that prints PEFT's true expected key set vs. the keys our rewrite produces — that diff is the next thing to look at. Full context, what's been tried, hardware notes for this 2×16 GB box, and the resume plan are in `examples/spraying/HANDOVER_INFERENCE.md`. Please read that document fully before running anything, then walk me through the resume steps one at a time.

---

## Background

### Project
- **Robot**: 8-DoF arm + binary sprayer trigger → 9-dim action.
  - Dims 0–7: joint deltas (normalized to [-1, 1] via q01/q99).
  - Dim 8: sprayer power (0/1, kept unnormalized — `mask[-1] = False`).
- **Task language**: `"spray the surface"` (single task).
- **Dataset**: `spraying-v1-cleaned` (449 episodes, 431k frames) ↦
  preprocessed to `spraying-v1-cleaned-processed` with depth tokens,
  visual trace, and action chunks.
- **Cameras**: base + wrist, 224×224 RGB.
- **Base model**: `allenai/MolmoAct-7B-D-0812` (Qwen2.5-7B based, bf16 ≈ 14 GB).
- **Adapter**: LoRA `rank=32 alpha=16 dropout=0.0`, target_modules ≈ all-linear
  across LLM blocks + ViT + connector. Saved at training-side path
  `checkpoints/molmoact_spraying_cleaned_lora_v1/step20000-lora/`.
  - `adapter_model.safetensors` ≈ 361 MB.

### Where the artifacts live
- **VM (`a100gpu8`)**: full repo, processed dataset at
  `~/data/molmoact/spraying-v1-cleaned-processed/`, checkpoints at
  `~/molmoact/checkpoints/molmoact_spraying_cleaned_lora_v1/`.
- **S3** (`vla-data-collection`): final LoRA adapter pushed there.
  Confirm exact prefix with the user — likely
  `s3://vla-data-collection/molmoact/molmoact_spraying_cleaned_lora_v1/step20000-lora/`.
- **Repo fork**: `git@github.com:faizan050903/molmoact.git`, branch
  `spraying-finetune`. Most recent commit during the inference debug:
  `75b9b19`.

---

## The current bug (the only thing left)

`examples/spraying/inference.py` builds an HF model + PEFT adapter, but
the adapter's contribution to forward is **zero**:

```
=== Adapter on/off comparison ===
  with adapter:    1338 chars, 226 tokens
  without adapter: 1338 chars, 226 tokens
  outputs identical (token IDs): True
  >>> LoRA contribution = 0. Loaded but not applied. Investigate scaling/wiring.
```

The model emits the **base LIBERO 7-dim flat action** `[t1,...,t7]` instead
of the trained nested 8-step chunk with 9-token inner lists
`[[t1,...,t9], [t1,...,t9], ...]`. So `parse_action(unnorm_key=...)` with
`only_len=9` returns `[]`.

### What we know about the cause

1. **`adapter_config.json` had FSDP-contaminated `target_modules`** because
   PEFT auto-discovered `"all-linear"` while the model was already
   FSDP-wrapped. The saved list contained
   `"_fsdp_wrapped_module.ff_out"` and `"ff_out._fsdp_wrapped_module"` —
   names that never match any module in an unwrapped HF model.
   `prepare_clean_adapter()` now scrubs those into bare names.

2. **`adapter_model.safetensors` keys are in training-time format**, which
   differs structurally from HF MolmoAct:
   - Training-time LLM block: `transformer.blocks.X.{att_proj,attn_out,ff_proj,ff_out}` (flat).
   - HF LLM block:             `transformer.blocks.X.{self_attn.att_proj, self_attn.attn_out, mlp.ff_proj, mlp.ff_out}`.
   - Training-time ViT block: probably `vision_backbone.image_vit.transformer.resblocks.X.{wq,wk,wv,wo,w1,w2}` (flat).
   - HF ViT block:            `vision_backbone.image_vit.transformer.resblocks.X.{attention.wq, ..., feed_forward.w1, feed_forward.w2}`.
   - Plus everything has `_fsdp_wrapped_module` / `_checkpoint_wrapped_module`
     wrappers sprinkled through, an extra `.model.` after `base_model.`
     because HF's `MolmoActForImageTextToText` nests the inner Molmo under
     `.model`, and PEFT uses `lora_{A,B}.default.weight` not
     `lora_{A,B}.weight`.

3. **Our remap claims 540/542 mapped, but PEFT still loads zero into the
   actual LoRA tensors.** Either we matched to wrong HF paths (so PEFT
   ignored our overwrites because the keys it expected never appeared in
   the file), or PEFT's expected names differ from what we computed.

### What's been tried, in order
- Commit `b8da269`: full key remap walking HF `named_modules` for valid
  linear paths, trying bare/`self_attn`/`mlp`/`attention`/`feed_forward`
  insertions for each saved key. Also scrubs adapter_config.json.
- Commit `bbe6512`: adds `--compare_no_adapter` and prints `lora_B`
  nonzero counts after attach.
- Commit `75b9b19`: **adds the diagnostic that's about to give us the
  answer** — at `[3a/5]` it walks the base model for `nn.Linear` modules
  whose last-segment name is in `target_modules`, builds PEFT's true
  expected key set, and diffs vs. what our rewrite wrote into the
  safetensors. Prints 8 examples of "expected but not produced" and
  "produced but not expected".

**The next thing to run is `75b9b19`. The diff it prints will tell us
exactly how to fix the rewrite.**

---

## Resume plan

### 0. Hardware adaptations for 2×16 GB
The current inference script uses `device_map="cuda"` which puts the
whole 14 GB bf16 model on a single GPU. On 2×16 GB you should be fine
either way:
- **Single GPU**: 14 GB model + activations + KV cache fits in 16 GB.
- **Sharded across both GPUs**: change `device_map="cuda"` →
  `device_map="auto"`. HF will split the model between the two cards.

If you OOM, fall back to 4-bit quantization:
```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForImageTextToText.from_pretrained(
    args.base_model, trust_remote_code=True,
    quantization_config=bnb, device_map="auto",
)
```
(`pip install bitsandbytes` first.) Note: quantizing the base while
LoRA-adaptering it is fine; just keep `torch_dtype=torch.bfloat16` off
the call.

### 1. Set up the environment
```bash
git clone git@github.com:faizan050903/molmoact.git
cd molmoact
git checkout spraying-finetune
python -m venv .venv && source .venv/bin/activate
pip install -e . --no-deps   # install repo
pip install transformers peft safetensors datasets pillow numpy
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install tensorflow-cpu   # MolmoAct HF processor uses TF for image ops
# Do NOT install torchao (peft requires >=0.16, repo has 0.10 pinned; uninstall if present)
```

### 2. Pull artifacts from S3
The user has `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` for bucket
`vla-data-collection`. Ask them to share the exact S3 prefixes if not in
this doc. Likely commands (verify path with user):
```bash
mkdir -p ~/data/molmoact/spraying-v1-cleaned-processed
aws s3 sync \
  s3://vla-data-collection/molmoact/spraying-v1-cleaned-processed/ \
  ~/data/molmoact/spraying-v1-cleaned-processed/
# Adapter only (~361 MB):
mkdir -p ~/molmoact-ckpt/step20000-lora
aws s3 sync \
  s3://vla-data-collection/molmoact/molmoact_spraying_cleaned_lora_v1/step20000-lora/ \
  ~/molmoact-ckpt/step20000-lora/
```
The minimum required files are:
- `<adapter_dir>/adapter_config.json`
- `<adapter_dir>/adapter_model.safetensors`
- `~/data/molmoact/spraying-v1-cleaned-processed/dataset_statistics.json`
The full processed dataset is only needed if you want to pull a sample
frame with ground-truth action via `--frame_index`. If you can't fetch
30 GB, use `--base_image / --wrist_image / --instruction` instead.

### 3. Run the probe (this is THE diagnostic step)
```bash
python examples/spraying/inference.py \
    --adapter_dir ~/molmoact-ckpt/step20000-lora \
    --norm_stats_path ~/data/molmoact/spraying-v1-cleaned-processed/dataset_statistics.json \
    --dataset_path ~/data/molmoact/spraying-v1-cleaned-processed \
    --frame_index 100 \
    --max_new_tokens 1024 \
    --compare_no_adapter 2>&1 | tee /tmp/infer_out.log
```
Critical blocks to read in the output:
- `[3a/5] Computing expected PEFT keys ...` — prints **matched / missing /
  extra** counts and 8 examples of each. **This tells you exactly how to
  fix the rewrite.**
- `lora_B nonzero count: X/Y` — confirms whether physical LoRA weights
  are nonzero post-load.
- `=== Adapter on/off comparison ===` — `outputs identical: True/False`.

### 4. Decide what to fix

| Probe output | Interpretation | Fix |
|---|---|---|
| `matched ≈ 0`, large `missing` and `extra` | Our HF paths are completely wrong. Examples of `extras` show what we produced; `missing` shows what PEFT wants. | Rewrite `remap_adapter_keys` based on the exact translation visible in those two lists. |
| `matched > 0`, small `missing` | Most keys land but some module names don't translate. Look at the `missing` examples. | Extend `WRAP_SEGMENTS` or handle the specific parent-path mismatch. |
| `matched == produced`, but `lora_B nonzero == 0` | Keys align, weights themselves are zero. | Re-examine training save — `lora_B` may not have been written. Verify by reading the raw safetensors with `safetensors.load_file` and checking norms. |
| `matched == produced`, `lora_B nonzero > 0`, but `outputs identical: True` | Weights loaded correctly but PEFT's `scaling` is zero or the adapter is being disabled silently. | Check `adapter_config.json` `lora_alpha` and `r`; check `model.peft_config["default"].scaling`. |

### 5. Verify the fix
After patching, the same run should show:
- `matched == 540`, `missing == 0`, `extra ≤ 2` (the two spurious
  `transformer.ff_out` keys without block index are OK to leave extra).
- `lora_B nonzero count: 540/540` (or whatever the matched count is).
- `outputs identical (token IDs): False`.
- Generated text after the cue contains a nested 8-step chunk like
  `[[t1,...,t9], [t1,...,t9], ..., [t1,...,t9]]`.
- `parse_action(...)` returns a list of length 8, each inner list of
  length 9. `pred_first_step` populates and the per-dim error block
  prints values.

### 6. When it works
Push your fix on `spraying-finetune` and verify on a few different
`--frame_index` values (e.g., 100, 1000, 50000). Per-dim absolute errors
on dims 0–7 should be small (~0.01–0.1 in joint-delta units, depending
on the frame). Dim 8 (sprayer) should be 0 or 1, exactly matching ground
truth on most frames since it's a binary channel.

---

## Files you'll touch most
- `examples/spraying/inference.py` — the smoke-test script. The
  `remap_adapter_keys` function is what'll need fixing once the probe
  output is in hand.
- `examples/spraying/inspect_adapter.py` — standalone safetensors dumper.
  Useful if you want to look at the raw saved keys outside the main
  pipeline:
  `python examples/spraying/inspect_adapter.py <path-to-adapter_model.safetensors>`
- `olmo/hf_model/molmoact/modeling_molmoact.py` — the HF base model
  source. Read this if you need to confirm a module path's exact
  hierarchy. Relevant lines:
  - Line 867, 894 — LLM block attention linears (`att_proj`, `attn_out`).
  - Line 975, 976 — LLM block MLP linears (`ff_proj`, `ff_out`).
  - Line 998 — block wraps them as `self.self_attn = ...`.
  - Line 1002 — block wraps MLP as `self.mlp = ...`.
  - Line 314–332 — ViT attention (`wq/wk/wv/wo`).
  - Line 279–281 — ViT MLP (`w1/w2`).
  - Line 568–570 — image_projector MLP (`w1/w2/w3`).
  - Line 491 — `patch_embedding`.

## Files you should *not* touch
- `olmo/train/trainer.py`, `olmo/train/distributed_checkpointing.py` —
  the training-time save path. Already patched, training is done, don't
  re-debug.
- Anything in `preprocess/` — the dataset is already preprocessed and on
  S3.

## Useful prior knowledge
- LoRA tokens decode as non-ASCII Qwen2 high-vocab strings (the funny
  glyphs in the generated text — those are correct action tokens, not
  garbled). `parse_action`'s regex filters bracketed lists that have at
  least one non-ASCII character to discriminate action chunks from
  coordinate lists.
- `parse_action` expects `unnorm_key="spraying-v1-cleaned-processed"`
  (set as the script's default). The `inject_norm_stats` step copies
  `dataset_statistics.json` contents into `model.norm_stats` so the
  unnormalization lookup succeeds — that part already works.
- `parse_depth` and `parse_trace` already return sensible values in the
  current broken run, confirming generate/decoding work; only action
  parsing is gated on getting the LoRA contribution wired up.

Good luck. The probe at `[3a/5]` is doing most of the thinking — once
its output is in front of you, the fix is straightforward.
