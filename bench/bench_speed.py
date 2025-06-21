import argparse
import random
import time
import os, sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

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


def _bytes_for_state(n_qubits: int, dtype: torch.dtype = torch.complex64) -> int:
    """Сколько байт занимает statevector 2**n комплексных значений."""
    itemsize = 16 if dtype == torch.complex128 else 8  # complex128 = 16B, complex64 = 8B
    return (1 << n_qubits) * itemsize


def _cuda_mem_info(gpu_idx: int = 0):
    free, total = torch.cuda.mem_get_info(gpu_idx)
    return free, total


def bench(circ: QuantumCircuit, device: str, label: str, dtype: torch.dtype, force: bool):
    device_obj = torch.device(device)
    need_bytes = _bytes_for_state(circ.num_qubits, dtype)
    need_gib = need_bytes / 1024 ** 3

    if device_obj.type == "cuda":
        free, total = _cuda_mem_info(device_obj.index or 0)
        free_gib, total_gib = free / 1024 ** 3, total / 1024 ** 3
        print(f"[{label}] Требуется ~{need_gib:.2f} ГиБ; свободно {free_gib:.2f} / {total_gib:.2f} ГиБ на {device}")
        if not force and need_bytes * 1.1 > free:  # 10% запас
            print(f"[{label}] ⚠ Недостаточно свободной памяти – пропуск (use --force)\n")
            return label, None
    else:
        print(f"[{label}] Требуется ~{need_gib:.2f} ГиБ RAM (device={device})")

    start = time.time()
    circ.simulate(device=device_obj, dtype=dtype)
    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    dur = time.time() - start
    return label, dur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--qubits", type=int, default=20, help="количество кубитов")
    parser.add_argument("-d", "--depth", type=int, default=1024, help="глубина схемы")
    parser.add_argument("--devices", default="cpu,cuda", help="список устройств через запятую, напр. 'cpu,cuda:0'")
    parser.add_argument("--dtype", choices=["c64", "c128"], default="c64", help="complex dtype (c64/c128)")
    parser.add_argument("--force", action="store_true", help="игнорировать проверку памяти и запускать всё равно")
    args = parser.parse_args()
    circ = random_circuit(args.qubits, args.depth)
    dtype = torch.complex64 if args.dtype == "c64" else torch.complex128

    devices = [d.strip() for d in args.devices.split(',') if d.strip()]
    print(f"Qubits={args.qubits}, Depth={args.depth}, dtype={args.dtype}, devices={devices}\n")

    results = []
    for dev in devices:
        label = f"{dev}"
        res = bench(circ, dev, label, dtype, args.force)
        results.append(res)

    print("\n== Итоги ==")
    for lbl, t in results:
        if t is None:
            print(f"{lbl:12s}: -- пропущено --")
        else:
            print(f"{lbl:12s}: {t:.3f} s")

if __name__ == "__main__":
    main() 