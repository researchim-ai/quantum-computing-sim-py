from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .circuit import QuantumCircuit

__all__ = ["load_qasm"]


def _parse_line(line: str) -> tuple[str, list[int]] | None:
    # Удаляем комментарии
    line = line.split("//", 1)[0].strip()
    if not line or line.endswith(";") is False:
        return None
    line = line[:-1].strip()  # убираем ;
    if line.startswith("OPENQASM") or line.startswith("include") or line.startswith("qreg"):
        return None
    # примитивные гейты: h q[0]; cx q[0],q[1];
    if line.startswith("cx"):
        args = line[2:].strip()
        q1, q2 = (int(tok.split("[")[1].split("]")[0]) for tok in args.split(","))
        return ("cx", [q1, q2])
    else:
        name, arg = line.split()
        q = int(arg.split("[")[1].split("]")[0])
        return (name.lower(), [q])


def parse_qasm_str(src: str) -> QuantumCircuit:
    ops: list[tuple[str, list[int]]] = []
    for ln in src.splitlines():
        parsed = _parse_line(ln)
        if parsed:
            ops.append(parsed)
    # определяем число кубитов
    max_q = 0
    for _, qs in ops:
        max_q = max(max_q, max(qs))
    circ = QuantumCircuit(max_q + 1)
    for name, qs in ops:
        getattr(circ, name)(*qs)
    return circ


def load_qasm(path: str | Path) -> QuantumCircuit:
    text = Path(path).read_text()
    return parse_qasm_str(text) 