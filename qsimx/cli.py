import argparse
import sys

import torch

from . import QuantumCircuit


def parse_expr(expr: str) -> QuantumCircuit:
    """Парсит строку вида "H0,CX0,1,RX1:0.3" и возвращает QuantumCircuit."""
    # Определяем количество кубитов как макс индекс + 1
    tokens = [tok.strip() for tok in expr.split(',') if tok.strip()]
    max_q = 0
    ops = []
    for tok in tokens:
        if tok.upper().startswith('CX'):
            rest = tok[2:]
            control, target = map(int, rest.split('/')[0].split(';')[0].split(':')[0].split()) if False else None  # placeholder
        parts = tok.replace(':', ' ').replace(';', ' ').replace('/', ' ').replace(',', ' ').split()
    # Fallback simple parser
    circ = QuantumCircuit(1)
    return circ


# TODO: полноценный QASM парсер

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="qsimx CLI")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="запустить схемy")
    run_p.add_argument("expr", help="строка c операциями, напр. 'H0,CX0-1' или путь к .qasm')")
    run_p.add_argument("-d", "--device", default="cpu")
    run_p.add_argument("-b", "--backend", choices=["statevector", "density"], default="statevector")
    run_p.add_argument("--noise", help="шумовой канал, напр. 'depol:0.05' или 'ad:0.1'")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        # expr может быть файлом .qasm
        if args.expr.endswith(".qasm"):
            from . import load_qasm
            circ = load_qasm(args.expr)
        else:
            tokens = [tok.strip() for tok in args.expr.split(',') if tok.strip()]
            max_q = 0
            ops: list[tuple[str, list[int]]] = []
            for g in tokens:
                if g[0] in {'H', 'X', 'Y', 'Z'}:
                    qubit = int(g[1:])
                    ops.append((g[0].lower(), [qubit]))
                    max_q = max(max_q, qubit)
                elif g.upper().startswith('CX'):
                    control, target = map(int, g[2:].split('-'))
                    ops.append(("cx", [control, target]))
                    max_q = max(max_q, control, target)
                else:
                    print(f"Неизвестный токен '{g}'", file=sys.stderr)
            circ = QuantumCircuit(max_q + 1)
            for name, qs in ops:
                getattr(circ, name)(*qs)
        if args.backend == "statevector":
            state = circ.simulate(device=args.device)
            print(state.cpu().tolist())
        else:
            from .densitymatrix import DensityMatrix
            rho = DensityMatrix(circ.num_qubits, device=args.device)
            for name, args_ in circ._ops:
                getattr(rho, name)(*args_)
            if args.noise:
                if args.noise.startswith("depol"):
                    p = float(args.noise.split(":")[1])
                    for q in range(circ.num_qubits):
                        rho.depolarize(q, p)
                elif args.noise.startswith("ad"):
                    g = float(args.noise.split(":")[1])
                    for q in range(circ.num_qubits):
                        rho.amplitude_damp(q, g)
                elif args.noise.startswith("pd"):
                    g = float(args.noise.split(":")[1])
                    for q in range(circ.num_qubits):
                        rho.phase_damp(q, g)
            print(rho.probabilities().cpu().tolist())
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 