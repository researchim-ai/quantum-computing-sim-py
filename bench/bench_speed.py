import argparse, random, time
import torch
from qsimx import QuantumCircuit


def random_circuit(n_qubits: int, depth: int, cx_ratio: float = 0.2, seed: int = 0):
    rnd = random.Random(seed)
    circ = QuantumCircuit(n_qubits)
    one_q_gates = ["h", "x", "y", "z", "rx", "ry", "rz"]
    for _ in range(depth):
        if rnd.random() < cx_ratio:
            c, t = rnd.sample(range(n_qubits), 2)
            circ.cx(c, t)
        else:
            name = rnd.choice(one_q_gates)
            q = rnd.randrange(n_qubits)
            if name in {"rx", "ry", "rz"}:
                getattr(circ, name)(q, rnd.random() * 3.1415)
            else:
                getattr(circ, name)(q)
    return circ


def bench(circ: QuantumCircuit, device: str, label: str):
    start = time.time()
    circ.simulate(device=device)
    torch.cuda.synchronize() if torch.cuda.is_available() and device == "cuda" else None
    dur = time.time() - start
    return label, dur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--qubits", type=int, default=20)
    parser.add_argument("-d", "--depth", type=int, default=1024)
    args = parser.parse_args()
    circ = random_circuit(args.qubits, args.depth)
    results = []
    results.append(bench(circ, "cpu", "CPU-einsum"))
    if torch.cuda.is_available():
        results.append(bench(circ, "cuda", "CUDA-fast"))
    print("Qubits", args.qubits, "Depth", args.depth)
    for lbl, t in results:
        print(f"{lbl:12s}: {t:.3f} s")

if __name__ == "__main__":
    main() 