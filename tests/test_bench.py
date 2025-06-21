import pytest, torch, time
from qsimx import QuantumCircuit

def tiny_circ():
    circ = QuantumCircuit(20)
    for d in range(256):
        for q in range(20):
            circ.rx(q, 0.1*d)
        for i in range(0, 20, 2):
            circ.cx(i, i+1)
    return circ

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_faster_than_cpu():
    circ = tiny_circ()
    start = time.time()
    circ.simulate(device="cpu")
    cpu_t = time.time() - start
    torch.cuda.synchronize()
    start = time.time()
    circ.simulate(device="cuda")
    torch.cuda.synchronize()
    gpu_t = time.time() - start
    assert gpu_t < cpu_t * 0.5  # at least 2x speedup 