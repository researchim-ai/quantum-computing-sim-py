from __future__ import annotations

import torch
import math

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _single_kernel(
        state_ptr, gate_ptr, n_amplitudes, qubit_mask: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offset < n_amplitudes
        idx0 = tl.load(offset, mask=mask, other=0)
        idx1 = idx0 ^ qubit_mask
        # ensure idx0<idx1 to process each pair once
        process = idx0 < idx1
        idx0 = tl.where(process, idx0, -1)
        idx1 = tl.where(process, idx1, -1)
        amp0 = tl.load(state_ptr + idx0, mask=process)
        amp1 = tl.load(state_ptr + idx1, mask=process)
        g00 = tl.load(gate_ptr + 0)
        g01 = tl.load(gate_ptr + 1)
        g10 = tl.load(gate_ptr + 2)
        g11 = tl.load(gate_ptr + 3)
        new0 = g00 * amp0 + g01 * amp1
        new1 = g10 * amp0 + g11 * amp1
        tl.store(state_ptr + idx0, new0, mask=process)
        tl.store(state_ptr + idx1, new1, mask=process)

    def apply_single(state: torch.Tensor, gate: torch.Tensor, qubit: int):
        assert state.is_cuda and gate.is_cuda
        n = int(math.log2(state.numel()))
        mask = 1 << (n - qubit - 1)
        BLOCK = 1024
        n_blocks = (state.numel() + BLOCK - 1) // BLOCK
        _single_kernel[(n_blocks,)](
            state,
            gate.flatten(),
            state.numel(),
            mask,
            BLOCK=BLOCK,
        )

except ImportError:

    def apply_single(state: torch.Tensor, gate: torch.Tensor, qubit: int):  # type: ignore
        # Fallback to torch einsum
        n = int(torch.log2(torch.tensor(state.numel(), dtype=torch.float32)))
        dim_left = 1 << qubit
        dim_right = 1 << (n - qubit - 1)
        sv = state.view(dim_left, 2, dim_right)
        state.copy_(torch.einsum("ab, ibj -> iaj", gate, sv).reshape(-1)) 