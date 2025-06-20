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


def test_cz_phase_flip():
    sv = StateVector(2)
    sv.x(0)
    sv.x(1)  # подготавливаем |11>
    sv.cz(0, 1)
    # фаза -|11>
    expected = torch.tensor([0, 0, 0, -1], dtype=sv.dtype, device=sv.device)
    assert torch.allclose(sv.tensor, expected)


def test_swap_gate():
    sv = StateVector(2)
    sv.x(0)  # |10>
    sv.swap(0, 1)
    expected = torch.tensor([0, 1, 0, 0], dtype=sv.dtype, device=sv.device)  # |01>
    assert torch.allclose(sv.tensor, expected)


def test_u3_vs_rz_ry_rz():
    theta, phi, lam = 0.2, 0.4, -0.1
    sv1 = StateVector(1)
    sv1.u3(0, theta, phi, lam)

    sv2 = StateVector(1)
    sv2.rz(0, lam)
    sv2.ry(0, theta)
    sv2.rz(0, phi)

    assert torch.allclose(sv1.tensor, sv2.tensor, atol=1e-6)


def test_counts():
    sv = StateVector(2)
    sv.h(0)
    shots = 1000
    cnts = sv.counts(shots=shots)
    total = sum(cnts.values())
    assert total == shots
    # только биты 00 и 01 возможны (little-endian порядок)
    for bs in cnts:
        assert bs in {"00", "01"}

    # little-endian order; allow variability 

def test_circuit_sample_counts():
    circ = QuantumCircuit(2)
    circ.h(0)
    shots = 500
    samples = circ.sample(shots=shots)
    assert samples.shape == (shots, 2)
    cnts = circ.counts(shots=shots)
    assert sum(cnts.values()) == shots
    assert set(cnts.keys()).issubset({"00", "01"}) 

def test_pauli_expectations():
    sv = StateVector(1)
    sv.h(0)
    # |+> state
    assert torch.allclose(sv.exp_z(0), torch.tensor(0.0, dtype=torch.float32, device=sv.device))
    assert torch.allclose(sv.exp_x(0), torch.tensor(1.0, dtype=torch.float32, device=sv.device))
    sv.z(0)
    assert torch.allclose(sv.exp_x(0), torch.tensor(-1.0, dtype=torch.float32, device=sv.device)) 