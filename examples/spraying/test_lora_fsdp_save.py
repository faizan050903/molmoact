"""Reproduce the FSDP1 + PEFT sharded-save crash in <30 seconds.

Builds a tiny multi-block model, FSDP1-wraps it (matching production's
``by_block_and_size`` + ``use_orig_params=True``), PEFT-wraps it with LoRA,
then tries to save state via the molmoact checkpointer's ``_prepare_state_dict``.

Run with torchrun across multiple GPUs to actually trigger the FSDP hooks:

    cd ~/molmoact && source .venv/bin/activate
    NCCL_NET=Socket NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 \
        PYTHONPATH=. torchrun --nnodes=1 --nproc-per-node=4 \
        examples/spraying/test_lora_fsdp_save.py

Expected output on success:
    [rank 0] state_dict has N keys, top-level: ['model']
    [rank 0] sample key: ...
    [rank 0] PASS — checkpoint save path works with LoRA + FSDP1

On the buggy code (the unfixed _prepare_state_dict), this raises
AssertionError: FSDP assumes ...attn_out.weight is in the state_dict ...
"""

from __future__ import annotations

import functools
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


class TinyBlock(nn.Module):
    """A miniature transformer-ish block — multiple Linear layers PEFT will wrap."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.ff_proj = nn.Linear(dim, dim * 4, bias=False)
        self.ff_out = nn.Linear(dim * 4, dim, bias=False)
        self.att_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        x = self.attn_out(x)
        x = self.ff_out(torch.nn.functional.silu(self.ff_proj(x)))
        return self.att_proj(x)


class TinyModel(nn.Module):
    def __init__(self, n_blocks: int = 4, dim: int = 256):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    log = lambda s: print(f"[rank {rank}] {s}", flush=True)

    if rank == 0:
        log(f"world_size={world_size}, building model + FSDP + PEFT...")

    # Build model
    model = TinyModel(n_blocks=4, dim=256).to(local_rank)

    # FSDP1 wrap with NO_SHARD: each rank keeps a full replica, like DDP.
    # This avoids the broken FSDP gather code path that segfaults during
    # state_dict export when PEFT-wrapped layers are present.
    from torch.distributed.fsdp import ShardingStrategy

    auto_wrap = functools.partial(size_based_auto_wrap_policy, min_num_params=1024)
    fsdp_model = FSDP(
        model,
        use_orig_params=True,
        auto_wrap_policy=auto_wrap,
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )

    # PEFT LoRA wrap — same target as production: every Linear
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        init_lora_weights="gaussian",
    )
    peft_model = get_peft_model(fsdp_model, lora_cfg)
    if rank == 0:
        log("PEFT wrap done; trying to save state_dict...")

    # Use peft.get_peft_model_state_dict — iterates named_parameters() directly
    # instead of recursing through state_dict() (which triggers FSDP's broken hook).
    # With NO_SHARD sharding strategy, each rank has the full parameter data, so
    # rank 0's view is the complete LoRA state.
    log("Bypass: peft.get_peft_model_state_dict (no FSDP state_dict() call)...")
    from peft.utils.save_and_load import get_peft_model_state_dict

    try:
        lora_state = get_peft_model_state_dict(peft_model)
        if rank == 0:
            keys = list(lora_state.keys())
            log(f"OK: {len(keys)} LoRA keys")
            log(f"sample key: {keys[0] if keys else '<empty>'}")
            log(f"sample tensor shape: {lora_state[keys[0]].shape if keys else 'n/a'}")
            log("PASS — peft.get_peft_model_state_dict works with NO_SHARD FSDP")
    except BaseException as e:
        log(f"FAIL: {type(e).__name__}: {e}")
        raise

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
