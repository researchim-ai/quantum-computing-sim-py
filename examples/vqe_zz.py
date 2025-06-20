import math

import torch

from qsimx import QuantumCircuit


def energy(theta: torch.Tensor) -> torch.Tensor:
    """Вычислить ожидаемую энергию гамильтониана H = Z₀ Z₁."""
    circ = QuantumCircuit(2)
    circ.ry(0, theta[0])
    circ.ry(1, theta[1])
    circ.cx(0, 1)
    return circ.expect([("ZZ", [0, 1])])[0]


if __name__ == "__main__":
    torch.manual_seed(0)
    theta = torch.randn(2, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.1)

    for step in range(200):
        opt.zero_grad()
        loss = energy(theta)
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"step {step:3d}: energy={loss.item():.6f}, theta={theta.detach().cpu().numpy()}")

    print("Final energy:", energy(theta).item()) 