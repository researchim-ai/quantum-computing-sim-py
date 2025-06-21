import pytest, torch
from qsimx import QuantumCircuit

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_single_gate_matches_cpu():
    circ = QuantumCircuit(3)
    circ.h(0).rx(1, 0.3).rz(2, 1.1)
    state_gpu = circ.simulate(device="cuda")
    state_cpu = circ.simulate(device="cpu")
    assert torch.allclose(state_gpu.cpu(), state_cpu, atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_cx_matches_cpu():
    circ = QuantumCircuit(4)
    circ.h(0).cx(0,1).cx(2,3)
    state_gpu = circ.simulate(device="cuda")
    state_cpu = circ.simulate(device="cpu")
    assert torch.allclose(state_gpu.cpu(), state_cpu, atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_swap_matches_cpu():
    circ = QuantumCircuit(3)
    circ.x(0)
    circ.swap(0,2)
    state_gpu = circ.simulate(device="cuda")
    state_cpu = circ.simulate(device="cpu")
    assert torch.allclose(state_gpu.cpu(), state_cpu, atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_cz_matches_cpu():
    circ = QuantumCircuit(2)
    circ.x(0).x(1)
    circ.cz(0,1)
    state_gpu = circ.simulate(device="cuda")
    state_cpu = circ.simulate(device="cpu")
    assert torch.allclose(state_gpu.cpu(), state_cpu, atol=1e-5) 