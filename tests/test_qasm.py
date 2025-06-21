from qsimx import load_qasm
import torch

def test_qasm_bell():
    qasm = """OPENQASM 2.0;
qreg q[2];
h q[0];
cx q[0],q[1];
"""
    circ = load_qasm.__globals__['parse_qasm_str'](qasm)
    probs = circ.simulate().abs() ** 2
    expected = torch.tensor([0.5,0,0,0.5], dtype=probs.dtype, device=probs.device)
    assert torch.allclose(probs, expected, atol=1e-6) 