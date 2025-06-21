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
            circ = QuantumCircuit(0)
        for gate_tok in args.expr.split(','):
            g = gate_tok.strip()
            if not g:
                continue
            if g[0] in {'H', 'X', 'Y', 'Z'}:
                qubit = int(g[1:])
                if circ.num_qubits <= qubit:
                    circ = QuantumCircuit(qubit + 1).merge(circ) if hasattr(circ, 'merge') else QuantumCircuit(qubit+1)
                getattr(circ, g[0].lower())(qubit)
            elif g.upper().startswith('CX'):
                rest = g[2:]
                control, target = map(int, rest.split('-'))
                nqubits = max(control, target) + 1
                if circ.num_qubits < nqubits:
                    circ = QuantumCircuit(nqubits).merge(circ) if hasattr(circ, 'merge') else QuantumCircuit(nqubits)
                circ.cx(control, target)
            else:
                print(f"Неизвестный токен '{g}'", file=sys.stderr)
        if args.backend == "statevector":
            state = circ.simulate(device=args.device)
            print(state.cpu())
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
            print(rho.probabilities().cpu())
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 