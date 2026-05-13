"""Quick LoRA-adapter inference test for our trained MolmoAct spraying model.

Loads the HF base model `allenai/MolmoAct-7B-D-0812`, attaches our trained LoRA
adapter from a `step{N}-lora/` directory, and runs a single forward+generate
on two camera images with a task instruction. Prints the parsed action.

This bypasses the upstream merge_lora + convert_molmoact_to_hf path (which
requires loading the base model in dist_cp sharded format) and instead does
the equivalent via HuggingFace + PEFT directly.

Usage:
    python examples/spraying/inference.py \
        --adapter_dir checkpoints/molmoact_smoke_v4/step50-lora \
        --base_image /path/to/base_view.jpg \
        --wrist_image /path/to/wrist_view.jpg \
        --instruction "spray the surface"

NOTE: An adapter from only 50 training steps will produce mostly random output.
Use this script first to confirm the inference pipeline works end-to-end, then
re-run with the production checkpoint (step20000-lora) once training finishes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file


def strip_fsdp_prefixes(state_dict):
    """Rewrite adapter keys to drop FSDP/checkpoint wrapper prefixes.

    Our trained adapter was saved while the model was wrapped by FSDP, so keys
    look like:
        base_model.model._fsdp_wrapped_module.transformer.blocks.0._checkpoint_wrapped_module._fsdp_wrapped_module.att_proj.lora_A.weight

    HuggingFace's MolmoAct (no FSDP) expects:
        base_model.model.transformer.blocks.0.att_proj.lora_A.weight
    """
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("._fsdp_wrapped_module", "").replace("._checkpoint_wrapped_module", "")
        cleaned[new_k] = v
    return cleaned


def prepare_clean_adapter(adapter_dir: Path) -> Path:
    """Write a key-renamed copy of the adapter to a temp dir and return its path."""
    config_src = adapter_dir / "adapter_config.json"
    weights_src = adapter_dir / "adapter_model.safetensors"
    if not config_src.exists():
        sys.exit(f"adapter_config.json missing at {config_src}")
    if not weights_src.exists():
        sys.exit(f"adapter_model.safetensors missing at {weights_src}")

    tmp = Path(tempfile.mkdtemp(prefix="molmoact_adapter_clean_"))
    shutil.copy(config_src, tmp / "adapter_config.json")
    raw = load_file(str(weights_src))
    print(f"  raw adapter: {len(raw)} keys")
    print(f"    sample raw key: {next(iter(raw))}")
    clean = strip_fsdp_prefixes(raw)
    print(f"    sample clean key: {next(iter(clean))}")
    save_file(clean, str(tmp / "adapter_model.safetensors"))
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True,
                    help="Path to step{N}-lora/ produced by training.")
    ap.add_argument("--base_image", required=True, help="Primary camera image (224x224).")
    ap.add_argument("--wrist_image", required=True, help="Wrist camera image (224x224).")
    ap.add_argument("--instruction", default="spray the surface")
    ap.add_argument("--base_model", default="allenai/MolmoAct-7B-D-0812")
    ap.add_argument("--unnorm_key", default="spraying-v1-cleaned-processed",
                    help="Key in dataset_statistics.json for action de-normalization. "
                         "Defaults to our cleaned dataset name.")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    adapter_dir = Path(os.path.expanduser(args.adapter_dir))
    base_image_path = Path(os.path.expanduser(args.base_image))
    wrist_image_path = Path(os.path.expanduser(args.wrist_image))

    print(f"[1/4] Cleaning adapter keys (strip FSDP prefixes)...")
    clean_adapter_dir = prepare_clean_adapter(adapter_dir)
    print(f"  cleaned adapter at {clean_adapter_dir}")

    print(f"[2/4] Loading HF base model {args.base_model} and processor (bf16, cuda)...")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="left",
    )
    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    print(f"  base loaded, dtype={next(base.parameters()).dtype}")

    print(f"[3/4] Attaching LoRA adapter via PeftModel.from_pretrained...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(clean_adapter_dir))
    model.eval()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  adapter attached, {n_trainable:,} trainable params (LoRA)")

    print(f"[4/4] Building prompt + processing images...")
    prompt = (
        f"The task is {args.instruction}. "
        "What is the action that the robot should take. "
        f"To figure out the action that the robot should take to {args.instruction}, "
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
    imgs = [Image.open(p).convert("RGB") for p in [base_image_path, wrist_image_path]]
    inputs = processor(images=[imgs], text=text, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"Running generate(max_new_tokens={args.max_new_tokens})...")
    with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_tokens = generated_ids[:, inputs["input_ids"].size(1):]
    generated_text = processor.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f"\n=== Generated text (first 500 chars) ===")
    print(generated_text[:500])

    # Parse — these helpers live on the HF MolmoAct model class
    print(f"\n=== Parsed outputs ===")
    underlying = model.base_model.model if hasattr(model, "base_model") else model
    try:
        depth = underlying.parse_depth(generated_text)
        print(f"depth tokens parsed: {len(depth) if depth is not None else 'None'}")
    except Exception as e:
        print(f"parse_depth failed: {type(e).__name__}: {e}")
    try:
        trace = underlying.parse_trace(generated_text)
        print(f"trace parsed: {trace}")
    except Exception as e:
        print(f"parse_trace failed: {type(e).__name__}: {e}")
    try:
        action = underlying.parse_action(generated_text, unnorm_key=args.unnorm_key)
        print(f"action: {action}")
    except Exception as e:
        print(f"parse_action failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
