from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _cx_kernel(state_ptr, n_amp, control_mask: tl.constexpr, target_mask: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offset < n_amp
        idx = tl.load(offset, mask=mask, other=0)
        cond = (idx & control_mask) != 0
        cond &= (idx & target_mask) == 0
        idx_a = tl.where(cond, idx, -1)
        idx_b = idx_a ^ target_mask
        amp_a = tl.load(state_ptr + idx_a, mask=cond)
        amp_b = tl.load(state_ptr + idx_b, mask=cond)
        tl.store(state_ptr + idx_a, amp_b, mask=cond)
        tl.store(state_ptr + idx_b, amp_a, mask=cond)

    def apply_cx(state: torch.Tensor, control: int, target: int):
        n = int(math.log2(state.numel()))
        control_mask = 1 << (n - control - 1)
        target_mask = 1 << (n - target - 1)
        BLOCK = 1024
        n_blocks = (state.numel() + BLOCK - 1) // BLOCK
        _cx_kernel[(n_blocks,)](state, state.numel(), control_mask, target_mask, BLOCK=BLOCK)

except ImportError:

    def apply_cx(state: torch.Tensor, control: int, target: int):  # type: ignore
        # fallback executed in StateVector.cx already
        raise RuntimeError("Triton not available") 