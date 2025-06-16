from __future__ import annotations

from typing import Any, List, Tuple

import torch

from .statevector import StateVector

__all__ = ["QuantumCircuit"]


class QuantumCircuit:
    """Простейшая реализация квантовой схемы поверх ``StateVector``.

    Схема хранит список операций (название, *args), которые затем
    последовательно применяются к вектору состояния.
    """

    def __init__(self, num_qubits: int) -> None:
        if num_qubits <= 0:
            raise ValueError("num_qubits должно быть ≥ 1")
        self.num_qubits = num_qubits
        self._ops: List[Tuple[str, tuple[Any, ...]]] = []

    # ------------------------------------------------------------------
    # Добавление гейтов в схему
    # ------------------------------------------------------------------
    def h(self, qubit: int) -> "QuantumCircuit":
        self._ops.append(("h", (qubit,)))
        return self

    def x(self, qubit: int) -> "QuantumCircuit":
        self._ops.append(("x", (qubit,)))
        return self

    def z(self, qubit: int) -> "QuantumCircuit":
        self._ops.append(("z", (qubit,)))
        return self

    def rx(self, qubit: int, theta: float | torch.Tensor) -> "QuantumCircuit":
        self._ops.append(("rx", (qubit, theta)))
        return self

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        self._ops.append(("cx", (control, target)))
        return self

    # ------------------------------------------------------------------
    # Симуляция
    # ------------------------------------------------------------------
    def simulate(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Выполнить схему и вернуть итоговый ``torch.Tensor`` состояния.

        Возвращаемый тензор *разделяет* память с внутренним ``StateVector`` –
        изменение его приведёт к изменению состояния.
        """
        sv = StateVector(self.num_qubits, dtype=dtype, device=device)
        for name, args in self._ops:
            getattr(sv, name)(*args)  # динамический вызов метода по названию
        return sv.tensor

    # ------------------------------------------------------------------
    # Удобства
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        lines = [f"QuantumCircuit(num_qubits={self.num_qubits})"]
        for i, (name, args) in enumerate(self._ops):
            lines.append(f"  {i}: {name}{args}")
        return "\n".join(lines) 