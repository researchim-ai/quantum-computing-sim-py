from __future__ import annotations
import math
import torch
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _swap_kernel(state_ptr, n_amp, q1_mask: tl.constexpr, q2_mask: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offset < n_amp
        idx = tl.load(offset, mask=mask, other=0)
        bit1 = idx & q1_mask
        bit2 = idx & q2_mask
        cond = bit1 != bit2
        idx_a = tl.where(cond, idx, -1)
        idx_b = idx_a ^ (q1_mask | q2_mask)
        amp_a = tl.load(state_ptr + idx_a, mask=cond)
        amp_b = tl.load(state_ptr + idx_b, mask=cond)
        tl.store(state_ptr + idx_a, amp_b, mask=cond)
        tl.store(state_ptr + idx_b, amp_a, mask=cond)

    def apply_swap(state: torch.Tensor, q1: int, q2: int):
        n = int(math.log2(state.numel()))
        q1_mask = 1 << (n - q1 - 1)
        q2_mask = 1 << (n - q2 - 1)
        BLOCK = 1024
        n_blocks = (state.numel() + BLOCK - 1) // BLOCK
        _swap_kernel[(n_blocks,)](state, state.numel(), q1_mask, q2_mask, BLOCK=BLOCK)
except ImportError:
    def apply_swap(state: torch.Tensor, q1: int, q2: int):
        raise RuntimeError("Triton not available") 