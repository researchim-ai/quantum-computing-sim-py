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