import torch

from qsimx import QuantumCircuit


def energy(theta: torch.Tensor) -> torch.Tensor:
    circ = QuantumCircuit(2)
    circ.ry(0, theta[0])
    circ.ry(1, theta[1])
    circ.cx(0, 1)
    return circ.expect([("ZZ", [0, 1])])[0]


def test_vqe_zz():
    torch.manual_seed(0)
    theta = torch.randn(2, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        loss = energy(theta)
        loss.backward()
        opt.step()
    final_energy = energy(theta).item()
    assert final_energy < -0.9 