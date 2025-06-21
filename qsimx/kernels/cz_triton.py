from __future__ import annotations
import math
import torch
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _cz_kernel(state_ptr, n_amp, control_mask: tl.constexpr, target_mask: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offset < n_amp
        idx = tl.load(offset, mask=mask, other=0)
        cond = (idx & control_mask) != 0
        cond &= (idx & target_mask) != 0
        amp = tl.load(state_ptr + idx, mask=cond)
        tl.store(state_ptr + idx, -amp, mask=cond)

    def apply_cz(state: torch.Tensor, control: int, target: int):
        n = int(math.log2(state.numel()))
        control_mask = 1 << (n - control - 1)
        target_mask = 1 << (n - target - 1)
        BLOCK = 1024
        n_blocks = (state.numel() + BLOCK - 1) // BLOCK
        _cz_kernel[(n_blocks,)](state, state.numel(), control_mask, target_mask, BLOCK=BLOCK)
except ImportError:
    def apply_cz(state: torch.Tensor, control: int, target: int):
        raise RuntimeError("Triton not available") 