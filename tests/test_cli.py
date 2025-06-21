import subprocess, sys, json, os
from pathlib import Path

def test_cli_statevector(tmp_path):
    cmd = [sys.executable, '-m', 'qsimx.cli', 'run', 'H0,CX0-1']
    out = subprocess.check_output(cmd, text=True)
    lines = eval(out.strip())
    assert len(lines) == 4  # 2 qubits amplitudes


def test_cli_density_noise(tmp_path):
    qasm = tmp_path / 'bell.qasm'
    qasm.write_text('''OPENQASM 2.0;
qreg q[2];
h q[0];
cx q[0],q[1];
''')
    cmd = [sys.executable, '-m', 'qsimx.cli', 'run', str(qasm), '--backend', 'density', '--noise', 'depol:1.0']
    out = subprocess.check_output(cmd, text=True)
    probs = eval(out.strip())
    assert all(abs(p - 0.25) < 1e-3 for p in probs)


def test_cli_phase_damp(tmp_path):
    qasm = tmp_path / 'h1.qasm'
    qasm.write_text('''OPENQASM 2.0;
qreg q[1];
h q[0];
''')
    cmd = [sys.executable, '-m', 'qsimx.cli', 'run', str(qasm), '--backend', 'density', '--noise', 'pd:1.0']
    out = subprocess.check_output(cmd, text=True)
    probs = eval(out.strip())
    assert all(abs(p - 0.5) < 1e-3 for p in probs) 