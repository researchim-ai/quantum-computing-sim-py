import pytest, torch
from qsimx import QuantumCircuit

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_single_gate_matches_cpu():
    circ = QuantumCircuit(3)
    circ.h(0).rx(1, 0.3).rz(2, 1.1)
    state_gpu = circ.simulate(device="cuda")
    state_cpu = circ.simulate(device="cpu")
    assert torch.allclose(state_gpu.cpu(), state_cpu, atol=1e-5) 