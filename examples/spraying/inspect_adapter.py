"""Diagnostic: print raw keys from a saved LoRA adapter file.

Used to debug the "0 LoRA params" / "Found missing adapter keys" warning
from PEFT.from_pretrained — tells us what the actual saved format looks
like so inference.py's strip_fsdp_prefixes can be fixed to map correctly.

Usage:
    python examples/spraying/inspect_adapter.py \
        ~/molmoact/checkpoints/molmoact_spraying_cleaned_lora_v1/step20000-lora/adapter_model.safetensors
"""
import sys
from collections import Counter
from safetensors.torch import load_file


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: inspect_adapter.py <path-to-adapter_model.safetensors>")
    sd = load_file(sys.argv[1])
    keys = list(sd.keys())
    print(f"Total keys: {len(keys)}")

    print("\n--- First 8 keys (sorted) ---")
    for k in sorted(keys)[:8]:
        print(f"  {k}  [shape={tuple(sd[k].shape)}]")

    print("\n--- Keys mentioning transformer.blocks.0.self_attn.att_proj ---")
    hits = [k for k in keys if "transformer.blocks.0.self_attn.att_proj" in k]
    for k in hits[:8]:
        print(f"  {k}")
    if not hits:
        print("  (none — adapter targets different modules?)")

    print("\n--- Keys mentioning transformer.blocks.27 ---")
    hits = [k for k in keys if "transformer.blocks.27" in k]
    for k in hits[:8]:
        print(f"  {k}")

    print("\n--- Distinct first 2 dot-segments (top 15 by frequency) ---")
    prefixes = Counter('.'.join(k.split('.')[:2]) for k in keys)
    for p, n in prefixes.most_common(15):
        print(f"  {n:4d}× {p}")

    print("\n--- Substring scan ---")
    print(f"  with _fsdp_wrapped_module:        {sum('_fsdp_wrapped_module' in k for k in keys)}")
    print(f"  with _checkpoint_wrapped_module:  {sum('_checkpoint_wrapped_module' in k for k in keys)}")
    print(f"  starting 'base_model.model.':     {sum(k.startswith('base_model.model.') for k in keys)}")
    print(f"  starting 'transformer.':          {sum(k.startswith('transformer.') for k in keys)}")
    print(f"  containing '.lora_A.':            {sum('.lora_A.' in k for k in keys)}")
    print(f"  containing '.lora_B.':            {sum('.lora_B.' in k for k in keys)}")

    print("\n--- Distinct target module names (segment immediately before .lora_A.) ---")
    # E.g., from "...blocks.0.att_proj.lora_A.weight" extract "att_proj".
    targets = Counter()
    for k in keys:
        if ".lora_A." in k:
            seg = k.split(".lora_A.")[0].split(".")[-1]
            targets[seg] += 1
    for seg, n in targets.most_common():
        print(f"  {n:4d}× {seg}")

    print("\n--- 5 sample non-transformer-block keys (no 'transformer.blocks') ---")
    non_block = [k for k in keys if "transformer.blocks" not in k]
    print(f"  total non-block: {len(non_block)}")
    for k in sorted(set(non_block))[:8]:
        print(f"  {k}")

    print("\n--- Distinct parent paths for non-transformer-block keys ---")
    parent_paths = Counter()
    for k in non_block:
        if ".lora_A." in k or ".lora_B." in k:
            parent = k.replace("._fsdp_wrapped_module", "").replace("._checkpoint_wrapped_module", "")
            for marker in (".lora_A.", ".lora_B."):
                if marker in parent:
                    parent = parent.split(marker)[0]
                    break
            parent_paths[parent] += 1
    for p, n in parent_paths.most_common(20):
        print(f"  {n:4d}× {p}")


if __name__ == "__main__":
    main()
