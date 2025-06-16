from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch

__all__: Sequence[str] = ("StateVector",)


class StateVector:
    """Класс-обёртка над ``torch.Tensor`` для хранения и манипулирования
    квантовым вектором состояния.

    Параметры
    ----------
    num_qubits:
        Количество кубитов.
    dtype:
        Тип данных ``torch`` (по умолчанию ``torch.complex64``).
    device:
        Устройство PyTorch (``"cuda"`` или ``"cpu"``).
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if num_qubits <= 0:
            raise ValueError("num_qubits должно быть ≥ 1")
        self.num_qubits = num_qubits
        self.dtype: torch.dtype = dtype or torch.complex64
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        dim: int = 1 << num_qubits  # 2**n
        self.tensor: torch.Tensor = torch.zeros(dim, dtype=self.dtype, device=self.device)
        # |0...0⟩ состояние
        self.tensor[0] = 1.0 + 0.0j

    # ---------------------------------------------------------------------
    # Однокубитные гейты
    # ---------------------------------------------------------------------
    def _apply_single_qubit_gate(self, gate: torch.Tensor, qubit: int) -> None:
        """Применить произвольный 2×2 матричный гейт к ``qubit``.
        Это *in-place* операция над ``self.tensor``.
        """
        n = self.num_qubits
        if not (0 <= qubit < n):
            raise IndexError("Неверный индекс кубита")

        # Меняем shape: (..., 2, ...). Axis = 1 соответствует qubit.
        dim_left = 1 << qubit  # 2**qubit
        dim_right = 1 << (n - qubit - 1)
        sv = self.tensor.view(dim_left, 2, dim_right)
        # einsum: a,b -> new amplitude
        # gate @ [amp0, amp1]
        self.tensor = torch.einsum("ab, ibj -> iaj", gate.to(self.tensor), sv).reshape(-1)

    def h(self, qubit: int) -> None:
        """Hadamard гейт."""
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        gate = torch.tensor([[1, 1], [1, -1]], dtype=self.dtype, device=self.device) * inv_sqrt2
        self._apply_single_qubit_gate(gate, qubit)

    def x(self, qubit: int) -> None:
        gate = torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device)
        self._apply_single_qubit_gate(gate, qubit)

    def z(self, qubit: int) -> None:
        gate = torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device)
        self._apply_single_qubit_gate(gate, qubit)

    def rx(self, qubit: int, theta: torch.Tensor | float) -> None:
        """Поворот вокруг X-оси на угол ``theta`` (рад). Поддерживает ``theta``
        как ``float`` или ``torch.Tensor`` (для автоградиентов).
        """
        # Приведение к tensor на нужном устройстве / dtype
        theta_t = torch.as_tensor(theta, dtype=self.tensor.real.dtype, device=self.device)
        cos = torch.cos(theta_t / 2)
        isin = -1j * torch.sin(theta_t / 2)
        gate = torch.stack([torch.stack([cos, isin]), torch.stack([isin, cos])])
        gate = gate.to(self.tensor)
        self._apply_single_qubit_gate(gate, qubit)

    # ---------------------------------------------------------------------
    # Двухкубитные гейты
    # ---------------------------------------------------------------------
    def cx(self, control: int, target: int) -> None:
        """Контролируемый X (CNOT).

        Реализуем обмен амплитуд, где контрольный бит = 1.
        """
        if control == target:
            raise ValueError("control и target должны различаться")
        n = self.num_qubits
        if not (0 <= control < n and 0 <= target < n):
            raise IndexError("Неверный индекс кубита")

        # Маски битов
        control_mask = 1 << control
        target_mask = 1 << target
        dim = 1 << n
        idx = torch.arange(dim, device=self.device)
        cond = (idx & control_mask) != 0  # контрольный бит = 1
        idx_target0 = idx[cond & ((idx & target_mask) == 0)]
        idx_target1 = idx_target0 | target_mask  # переключаем target бит

        # Сwap амплитуд
        temp = self.tensor[idx_target0].clone()
        self.tensor[idx_target0] = self.tensor[idx_target1]
        self.tensor[idx_target1] = temp

    # ---------------------------------------------------------------------
    # API вспомогательные
    # ---------------------------------------------------------------------
    def probabilities(self) -> torch.Tensor:
        """Вернуть распределение вероятностей |ψ|² в виде 1-D тензора."""
        return self.tensor.abs() ** 2

    def measure_all(self) -> torch.Tensor:
        """Сэмплировать одно наблюдение по всем кубитам."""
        probs = self.probabilities()
        dist = torch.distributions.Categorical(probs)
        outcome = dist.sample()
        return outcome 