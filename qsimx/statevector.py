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

    def ry(self, qubit: int, theta: torch.Tensor | float) -> None:
        """Поворот вокруг Y-оси."""
        theta_t = torch.as_tensor(theta, dtype=self.tensor.real.dtype, device=self.device)
        cos = torch.cos(theta_t / 2)
        sin = torch.sin(theta_t / 2)
        gate = torch.stack(
            [torch.stack([cos, -sin]), torch.stack([sin, cos])]
        ).to(self.tensor)
        self._apply_single_qubit_gate(gate, qubit)

    def rz(self, qubit: int, theta: torch.Tensor | float) -> None:
        """Поворот вокруг Z-оси."""
        theta_t = torch.as_tensor(theta, dtype=self.tensor.real.dtype, device=self.device)
        phase = torch.exp(-0.5j * theta_t)
        gate = torch.diag(torch.stack([phase.conj(), phase]))  # [[e^{-iθ/2},0],[0,e^{iθ/2}]]
        gate = gate.to(self.tensor)
        self._apply_single_qubit_gate(gate, qubit)

    def y(self, qubit: int) -> None:
        """Pauli-Y."""
        gate = torch.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device)
        self._apply_single_qubit_gate(gate, qubit)

    def s(self, qubit: int) -> None:
        """Phase S = Rz(π/2)."""
        self.rz(qubit, math.pi / 2)

    def sdg(self, qubit: int) -> None:
        """S† = Rz(-π/2)."""
        self.rz(qubit, -math.pi / 2)

    def t(self, qubit: int) -> None:
        """T = Rz(π/4)."""
        self.rz(qubit, math.pi / 4)

    def tdg(self, qubit: int) -> None:
        """T† = Rz(-π/4)."""
        self.rz(qubit, -math.pi / 4)

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
    # Двухкубитные расширенные гейты
    # ---------------------------------------------------------------------
    def swap(self, qubit1: int, qubit2: int) -> None:
        """SWAP: обмен состояниями двух кубитов."""
        if qubit1 == qubit2:
            return
        n = self.num_qubits
        if not (0 <= qubit1 < n and 0 <= qubit2 < n):
            raise IndexError("Неверный индекс кубита")
        mask1 = 1 << qubit1
        mask2 = 1 << qubit2
        dim = 1 << n
        idx = torch.arange(dim, device=self.device)
        # индексы, где биты различаются (01)
        cond = ((idx & mask1) == 0) & ((idx & mask2) != 0)
        idx_a = idx[cond]
        idx_b = idx_a ^ (mask1 | mask2)
        temp = self.tensor[idx_a].clone()
        self.tensor[idx_a] = self.tensor[idx_b]
        self.tensor[idx_b] = temp

    def cz(self, control: int, target: int) -> None:
        """Контролируемый Z: умножает амплитуды |11⟩ на -1."""
        if control == target:
            raise ValueError("control и target должны различаться")
        n = self.num_qubits
        if not (0 <= control < n and 0 <= target < n):
            raise IndexError("Неверный индекс кубита")
        control_mask = 1 << control
        target_mask = 1 << target
        dim = 1 << n
        idx = torch.arange(dim, device=self.device)
        cond = ((idx & control_mask) != 0) & ((idx & target_mask) != 0)
        self.tensor[cond] *= -1

    # ---------------------------------------------------------------------
    # API вспомогательные
    # ---------------------------------------------------------------------
    def probabilities(self) -> torch.Tensor:
        """Вернуть распределение вероятностей |ψ|² в виде 1-D тензора."""
        return self.tensor.abs() ** 2

    def sample(self, shots: int = 1024) -> torch.Tensor:
        """Вернуть ``shots`` сэмплов измерений всех кубитов.

        Возвращает 1-D тензор целых чисел размера ``shots``.
        """
        probs = self.probabilities()
        dist = torch.distributions.Categorical(probs)
        return dist.sample((shots,))

    def measure_all(self) -> torch.Tensor:
        """Устаревший алиас к ``sample(shots=1)[0]``."""
        return self.sample(shots=1)[0]

    # ---------------------------------------------------------------------
    # Сэмплы и статистика
    # ---------------------------------------------------------------------
    def sample_bits(
        self,
        *,
        shots: int = 1024,
        qubits: Sequence[int] | None = None,
        bit_order: str = "little",
    ) -> torch.Tensor:
        """Сэмплировать ``shots`` измерений указанных кубитов.

        Параметры
        ----------
        shots:
            Количество сэмплов.
        qubits:
            Последовательность индексов кубитов (по умолчанию *все*).
        bit_order:
            ``"little"`` (LSB — qubit-0) или ``"big"``.

        Возвращает
        ---------
        torch.Tensor
            ``shots × len(qubits)`` тензор типа ``int64`` (биты 0/1).
        """
        full_samples = self.sample(shots=shots)
        if qubits is None:
            qubits = list(range(self.num_qubits))
        qubits = list(qubits)
        if bit_order == "big":
            qubits = qubits[::-1]
        bits = [(full_samples >> q) & 1 for q in qubits]
        return torch.stack(bits, dim=1)  # shape (shots, len(qubits))

    def counts(
        self,
        *,
        shots: int = 1024,
        qubits: Sequence[int] | None = None,
        bit_order: str = "little",
    ) -> dict[str, int]:
        """Возвращает словарь bitstring → частота."""
        bit_arr = self.sample_bits(shots=shots, qubits=qubits, bit_order=bit_order)
        bitstrings = ["".join(map(str, row.tolist())) for row in bit_arr]
        freq: dict[str, int] = {}
        for bs in bitstrings:
            freq[bs] = freq.get(bs, 0) + 1
        return freq

    # ---------------------------------------------------------------------
    # Универсальный однокубитный гейт U3
    # ---------------------------------------------------------------------
    def u3(
        self,
        qubit: int,
        theta: torch.Tensor | float,
        phi: torch.Tensor | float,
        lam: torch.Tensor | float,
    ) -> None:
        """U3(θ, φ, λ) = Rz(φ) · Ry(θ) · Rz(λ)."""
        self.rz(qubit, lam)
        self.ry(qubit, theta)
        self.rz(qubit, phi)

    # ---------------------------------------------------------------------
    # Ожидания операторов Паули
    # ---------------------------------------------------------------------
    def exp_z(self, qubit: int) -> torch.Tensor:
        """⟨Z₍q₎⟩."""
        mask = 1 << qubit
        idx = torch.arange(self.tensor.numel(), device=self.device)
        sign = 1 - 2 * ((idx & mask) != 0).to(self.tensor)
        return (sign * (self.tensor.abs() ** 2)).sum().real

    def exp_x(self, qubit: int) -> torch.Tensor:
        """⟨X₍q₎⟩ = 2·Re⟨ψ|0⟩⟨1|ψ⟩."""
        mask = 1 << qubit
        dim = self.tensor.numel()
        idx0 = torch.arange(dim >> 1, device=self.device)
        # embed idx0 into full index with bit 0 at qubit
        left_mask = mask - 1
        high_bits = idx0 & ~left_mask
        low_bits = idx0 & left_mask
        idx_a = (high_bits << 1) | low_bits  # bit 0 at qubit
        idx_b = idx_a | mask  # flipped qubit
        amp_a = self.tensor[idx_a]
        amp_b = self.tensor[idx_b]
        return 2 * torch.real((amp_a.conj() * amp_b).sum()).real

    def exp_y(self, qubit: int) -> torch.Tensor:
        """⟨Y₍q₎⟩ = 2·Im⟨ψ|1⟩⟨0|ψ⟩."""
        mask = 1 << qubit
        dim = self.tensor.numel()
        idx0 = torch.arange(dim >> 1, device=self.device)
        left_mask = mask - 1
        high_bits = idx0 & ~left_mask
        low_bits = idx0 & left_mask
        idx_a = (high_bits << 1) | low_bits
        idx_b = idx_a | mask
        amp_a = self.tensor[idx_a]
        amp_b = self.tensor[idx_b]
        # ⟨Y⟩ = 2·Im(⟨a|b⟩)  with phase 1
        return 2 * torch.imag((amp_a.conj() * amp_b).sum()).real

    def exp_z_string(self, qubits: Sequence[int]) -> torch.Tensor:
        """Ожидание тензорного произведения ZᵢZⱼ…"""
        mask = 0
        for q in qubits:
            mask |= 1 << q
        idx = torch.arange(self.tensor.numel(), device=self.device)
        parity = ((idx & mask).bit_count() % 2 == 1).to(self.tensor)  # 1 если нечётное число 1
        sign = 1 - 2 * parity
        return (sign * (self.tensor.abs() ** 2)).sum().real 