import torch
from qsimx import DensityMatrix, StateVector


def test_density_pure_state_matches_statevector():
    sv = StateVector(2)
    sv.h(0)
    sv.cx(0, 1)
    rho = DensityMatrix(2)
    rho.h(0)
    rho.cx(0, 1)
    probs_sv = sv.probabilities()
    probs_rho = rho.probabilities()
    assert torch.allclose(probs_sv, probs_rho, atol=1e-6)
    assert torch.allclose(rho.trace(), torch.tensor(1.0))


def test_depolarize_channel():
    rho = DensityMatrix(1)
    rho.depolarize(0, p=1.0)
    expected = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=rho.dtype, device=rho.device)
    assert torch.allclose(rho.tensor, expected, atol=1e-6)


def test_amplitude_damp():
    rho = DensityMatrix(1)
    rho.x(0)  # подготовить |1><1|
    rho.amplitude_damp(0, gamma=1.0)
    expected = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=rho.dtype, device=rho.device)
    assert torch.allclose(rho.tensor, expected, atol=1e-6)


def test_phase_damp():
    rho = DensityMatrix(1)
    rho.h(0)  # состояние |+⟩
    rho.phase_damp(0, gamma=1.0)
    expected = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=rho.dtype, device=rho.device)
    assert torch.allclose(rho.tensor, expected, atol=1e-6)


def test_apply_kraus_identity():
    rho = DensityMatrix(1)
    rho.x(0)
    before = rho.tensor.clone()
    eye = torch.eye(2, dtype=rho.dtype, device=rho.device)
    rho.apply_kraus(0, [eye])
    assert torch.allclose(rho.tensor, before, atol=1e-6)


def test_bit_flip():
    rho = DensityMatrix(1)
    # начальное |0><0|
    rho.bit_flip(0, p=1.0)
    expected = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=rho.dtype, device=rho.device)
    assert torch.allclose(rho.tensor, expected, atol=1e-6)


def test_phase_flip():
    rho = DensityMatrix(1)
    rho.h(0)
    rho.phase_flip(0, p=1.0)
    expected = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=rho.dtype, device=rho.device)
    assert torch.allclose(rho.tensor, expected, atol=1e-6)


def test_y_flip():
    rho = DensityMatrix(1)
    rho.h(0)
    rho.y_flip(0, p=1.0)
    expected = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=rho.dtype, device=rho.device)
    assert torch.allclose(rho.tensor, expected, atol=1e-6) 