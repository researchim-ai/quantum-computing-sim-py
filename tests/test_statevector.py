import math

import torch
import pytest

from qsimx import QuantumCircuit, StateVector


def test_bell_state_probabilities():
    circ = QuantumCircuit(2)
    circ.h(0).cx(0, 1)
    probs = circ.simulate().abs() ** 2
    expected = torch.tensor([0.5, 0.0, 0.5, 0.0], dtype=probs.dtype, device=probs.device)
    assert torch.allclose(probs, expected, atol=1e-6)


def test_rx_gradient():
    theta = torch.tensor(0.3, requires_grad=True)
    sv = StateVector(1)
    sv.rx(0, theta)
    prob1 = sv.probabilities()[1]
    prob1.backward()
    expected_grad = 0.5 * math.sin(0.3)
    assert torch.allclose(theta.grad, torch.tensor(expected_grad, device=theta.grad.device), atol=1e-6)


def test_sample_shots_shape():
    sv = StateVector(3)
    sv.h(0)
    shots = 128
    samples = sv.sample(shots=shots)
    assert samples.shape == (shots,)
    assert samples.dtype in (torch.int64, torch.long)
    assert samples.min() >= 0 and samples.max() < 8  # 2**3 = 8 