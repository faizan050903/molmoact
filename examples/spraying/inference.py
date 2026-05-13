"""LoRA-adapter inference test for our trained MolmoAct spraying model.

Loads the HF base model `allenai/MolmoAct-7B-D-0812`, attaches our trained LoRA
adapter from a `step{N}-lora/` directory, injects the spraying-dataset
norm_stats so `parse_action` can de-normalize into real units, runs a single
generate, and prints the parsed action alongside the dataset's ground-truth
action for that frame.

Two ways to specify the inputs:

A) From the processed dataset by frame index (recommended — also gives a
   ground-truth action to compare against):

    python examples/spraying/inference.py \
        --adapter_dir checkpoints/molmoact_spraying_cleaned_lora_v1/step20000-lora \
        --dataset_path ~/data/molmoact/spraying-v1-cleaned-processed \
        --frame_index 100

B) From standalone image files (no ground truth):

    python examples/spraying/inference.py \
        --adapter_dir checkpoints/molmoact_spraying_cleaned_lora_v1/step20000-lora \
        --base_image /path/to/base.jpg \
        --wrist_image /path/to/wrist.jpg \
        --instruction "spray the surface"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file


def strip_fsdp_prefixes(state_dict):
    """Rewrite adapter keys to match the HF model's PEFT layout.

    Training-time wrapping (saved into the adapter):
        base_model.model._fsdp_wrapped_module.<X>._checkpoint_wrapped_module._fsdp_wrapped_module.<Y>.lora_A.default.weight

    HF-time PEFT layout (what PeftModel.from_pretrained expects):
        base_model.model.model.<X>.<Y>.lora_A.default.weight

    Two transforms:
    1. Drop `._fsdp_wrapped_module` and `._checkpoint_wrapped_module` (FSDP wraps).
    2. Insert an extra `.model.` after `base_model.model.` — because at HF
       inference, `PeftModel.base_model.model` is the HuggingFace
       `MolmoActForImageTextToText` wrapper which itself contains `.model` (the
       inner Molmo), whereas at training time PEFT wrapped the inner Molmo
       directly. So the path has one extra `.model.` in the HF case.
    """
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("._fsdp_wrapped_module", "").replace("._checkpoint_wrapped_module", "")
        if new_k.startswith("base_model.model.") and not new_k.startswith("base_model.model.model."):
            new_k = "base_model.model.model." + new_k[len("base_model.model."):]
        cleaned[new_k] = v
    return cleaned


def prepare_clean_adapter(adapter_dir: Path) -> Path:
    config_src = adapter_dir / "adapter_config.json"
    weights_src = adapter_dir / "adapter_model.safetensors"
    if not config_src.exists():
        sys.exit(f"adapter_config.json missing at {config_src}")
    if not weights_src.exists():
        sys.exit(f"adapter_model.safetensors missing at {weights_src}")
    tmp = Path(tempfile.mkdtemp(prefix="molmoact_adapter_clean_"))
    shutil.copy(config_src, tmp / "adapter_config.json")
    raw = load_file(str(weights_src))
    clean = strip_fsdp_prefixes(raw)
    save_file(clean, str(tmp / "adapter_model.safetensors"))
    print(f"  cleaned adapter at {tmp} ({len(clean)} keys)")
    return tmp


def find_norm_stats_owner(model):
    """Walk the (possibly PEFT-wrapped) model to find the module that owns
    a `norm_stats` attribute. Returns the inner MolmoAct model object."""
    # PeftModel → LoraModel → MolmoActForImageTextToText
    candidates = [model]
    for attr in ("base_model", "model"):
        if candidates and hasattr(candidates[-1], attr):
            candidates.append(getattr(candidates[-1], attr))
    for m in reversed(candidates):
        if hasattr(m, "norm_stats"):
            return m
    # Fallback: scan all submodules
    for m in model.modules():
        if hasattr(m, "norm_stats"):
            return m
    raise RuntimeError("Could not find `norm_stats` attribute on model or any submodule")


def inject_norm_stats(model, stats_path: Path):
    """Load dataset_statistics.json and merge it into the model's norm_stats dict.
    Returns the list of keys now present (for printing)."""
    with open(stats_path) as f:
        stats = json.load(f)
    owner = find_norm_stats_owner(model)
    if not isinstance(owner.norm_stats, dict):
        owner.norm_stats = {}
    for k, v in stats.items():
        owner.norm_stats[k] = v
    return list(owner.norm_stats.keys())


def load_frame_from_dataset(dataset_path: Path, frame_index: int):
    """Pull image + wrist_image + ground-truth action + language from the
    processed LeRobot/HF dataset at the given frame index. Returns dict."""
    from datasets import load_from_disk
    ds = load_from_disk(str(dataset_path))
    if frame_index < 0 or frame_index >= len(ds):
        sys.exit(f"frame_index {frame_index} out of range (dataset has {len(ds)} frames)")
    sample = ds[frame_index]
    return {
        "base_image": sample["image"] if hasattr(sample["image"], "convert") else Image.open(sample["image"]).convert("RGB"),
        "wrist_image": sample["wrist_image"] if hasattr(sample["wrist_image"], "convert") else Image.open(sample["wrist_image"]).convert("RGB"),
        "instruction": sample.get("language_instruction", "spray the surface"),
        "gt_action": sample.get("actions"),
        "gt_state": sample.get("state"),
        "episode": sample.get("episode_index"),
        "frame": sample.get("frame_index"),
    }


def fmt_action(action, action_dim_labels):
    """Pretty-print an action vector (list-of-floats) alongside dim labels."""
    if action is None:
        return "  <none>"
    if hasattr(action, "tolist"):
        action = action.tolist()
    lines = []
    for i, (label, v) in enumerate(zip(action_dim_labels, action)):
        lines.append(f"    [{i}] {label:14s} = {float(v):+.5f}")
    return "\n".join(lines)


SPRAYING_ACTION_LABELS = [
    "liftkit_mid", "liftkit_top", "shoulder_pan", "shoulder_lift",
    "elbow", "wrist_1", "wrist_2", "wrist_3", "sprayer_pwr",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True,
                    help="Path to step{N}-lora/ produced by training.")
    # Mode A: from dataset (preferred — gives a ground-truth comparison)
    ap.add_argument("--dataset_path",
                    default=os.path.expanduser("~/data/molmoact/spraying-v1-cleaned-processed"),
                    help="Path to the processed LeRobot/HF dataset.")
    ap.add_argument("--frame_index", type=int, default=None,
                    help="Frame index in --dataset_path. If set, overrides --base_image/--wrist_image.")
    # Mode B: standalone files
    ap.add_argument("--base_image", help="Primary camera image path.")
    ap.add_argument("--wrist_image", help="Wrist camera image path.")
    ap.add_argument("--instruction", default="spray the surface")
    # Stats + model knobs
    ap.add_argument("--norm_stats_path",
                    default=os.path.expanduser("~/data/molmoact/spraying-v1-cleaned-processed/dataset_statistics.json"),
                    help="Path to dataset_statistics.json from preprocessing.")
    ap.add_argument("--unnorm_key", default="spraying-v1-cleaned-processed",
                    help="Key in dataset_statistics.json for action de-normalization.")
    ap.add_argument("--base_model", default="allenai/MolmoAct-7B-D-0812")
    # Depth alone can fill ~256 tokens before trace+action are emitted, so 1024
    # is the safe floor for diagnosing whether the action chunk is generated.
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--print_full_text", action="store_true",
                    help="Print the entire generated string (not the 800-char head).")
    args = ap.parse_args()

    adapter_dir = Path(os.path.expanduser(args.adapter_dir))

    # --- Pick input source ---
    gt_action = None
    if args.frame_index is not None:
        print(f"[setup] Loading frame {args.frame_index} from {args.dataset_path}")
        frame = load_frame_from_dataset(Path(args.dataset_path), args.frame_index)
        base_image = frame["base_image"]
        wrist_image = frame["wrist_image"]
        instruction = frame["instruction"]
        gt_action = frame["gt_action"]
        print(f"  episode={frame['episode']}, frame={frame['frame']}, instruction={instruction!r}")
        if gt_action is not None:
            print(f"  ground-truth action (this frame): {[round(float(x), 5) for x in (gt_action.tolist() if hasattr(gt_action, 'tolist') else gt_action)]}")
    else:
        if not (args.base_image and args.wrist_image):
            sys.exit("Either pass --frame_index, or --base_image AND --wrist_image.")
        base_image = Image.open(os.path.expanduser(args.base_image)).convert("RGB")
        wrist_image = Image.open(os.path.expanduser(args.wrist_image)).convert("RGB")
        instruction = args.instruction
        print(f"[setup] Using standalone images, instruction={instruction!r}")

    # --- Clean adapter ---
    print(f"[1/5] Cleaning adapter keys...")
    clean_adapter_dir = prepare_clean_adapter(adapter_dir)

    # --- Load base ---
    print(f"[2/5] Loading HF base {args.base_model} (bf16, cuda)...")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="left",
    )
    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )

    # --- Attach LoRA ---
    print(f"[3/5] Attaching LoRA adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(clean_adapter_dir))
    model.eval()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  adapter attached, {n_trainable:,} LoRA params")

    # --- Inject spraying norm_stats ---
    print(f"[4/5] Injecting norm_stats from {args.norm_stats_path}...")
    keys_after = inject_norm_stats(model, Path(args.norm_stats_path))
    print(f"  model.norm_stats keys: {keys_after}")
    if args.unnorm_key not in keys_after:
        sys.exit(f"--unnorm_key {args.unnorm_key!r} not found after injection. Available: {keys_after}")

    # --- Build prompt + process inputs ---
    print(f"[5/5] Building prompt + processing images...")
    prompt = (
        f"The task is {instruction}. "
        "What is the action that the robot should take. "
        f"To figure out the action that the robot should take to {instruction}, "
        "let's think through it step by step. "
        "First, what is the depth map for the first image? "
        "Second, what is the trajectory of the end effector in the first image? "
        "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
        "along with other images from different camera views as additional information, "
        "what is the action that the robot should take?"
    )
    text = processor.apply_chat_template(
        [{"role": "user", "content": [dict(type="text", text=prompt)]}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(images=[[base_image, wrist_image]], text=text, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"Running generate(max_new_tokens={args.max_new_tokens})...")
    with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_tokens = generated_ids[:, inputs["input_ids"].size(1):]
    generated_text = processor.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    n_generated = generated_tokens.size(1)
    print(f"\n=== Generated text ({len(generated_text)} chars, {n_generated} tokens) ===")
    if args.print_full_text or len(generated_text) <= 1200:
        print(generated_text)
    else:
        print(generated_text[:600])
        print("\n  ...[middle elided]...\n")
        print(generated_text[-600:])

    # Locate the action-emission cue so we can confirm whether the model got
    # past depth+trace into the action phase.
    cue = "the action that the robot should take is"
    cue_idx = generated_text.lower().find(cue)
    if cue_idx >= 0:
        print(f"\n[diag] Found cue {cue!r} at char {cue_idx}; following 400 chars:")
        print(generated_text[cue_idx:cue_idx + 400])
    else:
        print(f"\n[diag] Cue {cue!r} NOT found — model didn't reach the action phase "
              f"within {args.max_new_tokens} tokens, or trained format differs.")

    print(f"\n=== Parsed outputs ===")
    underlying = find_norm_stats_owner(model)
    try:
        depth = underlying.parse_depth(generated_text)
        n_depth = len(depth) if depth is not None else 0
        print(f"depth tokens parsed: {n_depth}")
    except Exception as e:
        print(f"parse_depth failed: {type(e).__name__}: {e}")
    try:
        trace = underlying.parse_trace(generated_text)
        print(f"trace points: {trace}")
    except Exception as e:
        print(f"parse_trace failed: {type(e).__name__}: {e}")

    pred_action = None
    try:
        pred_action = underlying.parse_action(generated_text, unnorm_key=args.unnorm_key)
        print(f"parsed action (unnormalized): {pred_action}")
    except Exception as e:
        print(f"parse_action failed: {type(e).__name__}: {e}")

    # --- Comparison ---
    print(f"\n=== Action comparison (in physical units after de-normalization) ===")
    print(f"Spraying action dims: {SPRAYING_ACTION_LABELS}")

    # Note: parse_action may return a single action (9-dim) or a chunk (8 × 9).
    # Try to handle both shapes.
    pred_first_step = None
    if pred_action is not None:
        if hasattr(pred_action, "tolist"):
            pred_action = pred_action.tolist()
        # If it's a list of lists (chunk), take the first step
        if isinstance(pred_action, list) and pred_action and isinstance(pred_action[0], list):
            print(f"\nPredicted action chunk shape: {len(pred_action)} steps x {len(pred_action[0])} dims")
            pred_first_step = pred_action[0]
            print(f"\nPredicted (first step of chunk):")
            print(fmt_action(pred_first_step, SPRAYING_ACTION_LABELS))
        else:
            pred_first_step = pred_action
            print(f"\nPredicted (single step):")
            print(fmt_action(pred_first_step, SPRAYING_ACTION_LABELS))

    if gt_action is not None:
        print(f"\nGround truth (from dataset frame):")
        print(fmt_action(gt_action, SPRAYING_ACTION_LABELS))

    if gt_action is not None and pred_first_step is not None:
        gt_list = gt_action.tolist() if hasattr(gt_action, "tolist") else list(gt_action)
        print(f"\nPer-dim absolute error |pred - gt|:")
        for i, (label, p, g) in enumerate(zip(SPRAYING_ACTION_LABELS, pred_first_step, gt_list)):
            err = abs(float(p) - float(g))
            print(f"    [{i}] {label:14s} pred={float(p):+.5f}  gt={float(g):+.5f}  |err|={err:.5f}")


if __name__ == "__main__":
    main()
