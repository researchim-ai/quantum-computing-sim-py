from __future__ import annotations

import math
from typing import Sequence

import torch

__all__ = ["DensityMatrix"]


class DensityMatrix:
    """Простейшая реализация density matrix ρ размера 2ⁿ×2ⁿ."""

    def __init__(
        self,
        num_qubits: int,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        autorescale: bool = False,
    ) -> None:
        if num_qubits <= 0:
            raise ValueError("num_qubits ≥ 1")
        self.num_qubits = num_qubits
        self.dtype = dtype or torch.complex64
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.autorescale = autorescale
        dim = 1 << num_qubits
        self.tensor = torch.zeros((dim, dim), dtype=self.dtype, device=self.device)
        self.tensor[0, 0] = 1.0 + 0j  # |0><0|

    # ------------------------------------------------------------------
    # Однокубитные унитарные гейты
    # ------------------------------------------------------------------
    def _apply_unitary_single(self, U: torch.Tensor, qubit: int):
        # Строим полный U ⊗ I через крон-произведение.
        mats: list[torch.Tensor] = []
        for q in range(self.num_qubits):
            if q == qubit:
                mats.append(U.to(self.tensor))
            else:
                mats.append(torch.eye(2, dtype=self.dtype, device=self.device))
        # Крон: I ⊗ ... ⊗ U ⊗ ...
        U_full = mats[0]
        for m in mats[1:]:
            U_full = torch.kron(U_full, m)
        self.tensor = U_full @ self.tensor @ U_full.conj().T
        self._maybe_rescale()

    def h(self, qubit: int):
        inv = 1 / math.sqrt(2)
        U = torch.tensor([[1, 1], [1, -1]], dtype=self.dtype, device=self.device) * inv
        self._apply_unitary_single(U, qubit)

    def x(self, qubit: int):
        U = torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device)
        self._apply_unitary_single(U, qubit)

    def z(self, qubit: int):
        U = torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device)
        self._apply_unitary_single(U, qubit)

    # ------------------------------------------------------------------
    # Двухкубитный CX
    # ------------------------------------------------------------------
    def cx(self, control: int, target: int):
        n = self.num_qubits
        if control == target:
            raise ValueError
        dim = 1 << n
        idx = torch.arange(dim, device=self.device)
        control_mask = 1 << control
        target_mask = 1 << target
        # строим унитарно: если control=1, флип target
        U = torch.eye(dim, dtype=self.dtype, device=self.device)
        cond = (idx & control_mask) != 0
        rows = idx[cond & ((idx & target_mask) == 0)]
        cols = rows | target_mask
        U[rows, rows] = 0
        U[cols, cols] = 0
        U[rows, cols] = 1
        U[cols, rows] = 1
        # ρ -> U ρ U†
        self.tensor = U @ self.tensor @ U.conj().T
        self._maybe_rescale()

    # ------------------------------------------------------------------
    # Шум: деполяризующий канал
    # ------------------------------------------------------------------
    def depolarize(self, qubit: int, p: float):
        """Деполяризующий канал: ρ -> (1-p)ρ + p * I/2."""
        if not (0 <= p <= 1):
            raise ValueError
        n = self.num_qubits
        dim = 1 << n
        I = torch.eye(dim, dtype=self.dtype, device=self.device)
        # среднее по Паули I,X,Y,Z = 1/4 Σ σ ρ σ
        # Здесь берём простую I/2 на выбранный кубит → тензорное I/2 ⊗ I_rest.
        mask = 1 << qubit
        mixed = self._partial_mixed(qubit)
        self.tensor = (1 - p) * self.tensor + p * mixed
        self._maybe_rescale()

    def _partial_mixed(self, qubit: int) -> torch.Tensor:
        """Возвращает состояние I/2 на qubit ⊗ I_rest/2^{n-1}."""
        dim = 1 << self.num_qubits
        return torch.eye(dim, dtype=self.dtype, device=self.device) / dim

    # ------------------------------------------------------------------
    # Метрики
    # ------------------------------------------------------------------
    def trace(self) -> torch.Tensor:
        return torch.real(torch.trace(self.tensor))

    def probabilities(self) -> torch.Tensor:
        return torch.real(torch.diag(self.tensor))

    def amplitude_damp(self, qubit: int, gamma: float):
        """Амплитудная релаксация (T1): параметр ``gamma`` = 1-exp(-t/T1)."""
        if not (0 <= gamma <= 1):
            raise ValueError
        g = torch.tensor(gamma, dtype=self.tensor.real.dtype, device=self.device)
        k0 = torch.tensor([[1.0, 0.0], [0.0, torch.sqrt(1 - g)]], dtype=self.dtype, device=self.device)
        k1 = torch.tensor([[0.0, torch.sqrt(g)], [0.0, 0.0]], dtype=self.dtype, device=self.device)
        kraus = [k0, k1]
        new_rho = torch.zeros_like(self.tensor)
        for K in kraus:
            mats = []
            for q in range(self.num_qubits):
                mats.append(K if q == qubit else torch.eye(2, dtype=self.dtype, device=self.device))
            U = mats[0]
            for m in mats[1:]:
                U = torch.kron(U, m)
            new_rho += U @ self.tensor @ U.conj().T
        self.tensor = new_rho
        self._maybe_rescale()

    # ------------------------------------------------------------------
    # Шум: фазовая релаксация
    # ------------------------------------------------------------------
    def phase_damp(self, qubit: int, gamma: float):
        """Фазовая релаксация (T₂ / phase damping).  Параметр ``gamma`` — вероятность потери когерентности (0 ≤ γ ≤ 1)."""
        if not (0 <= gamma <= 1):
            raise ValueError
        g = torch.tensor(gamma, dtype=self.tensor.real.dtype, device=self.device)
        sqrt_g = torch.sqrt(g)
        sqrt_1mg = torch.sqrt(1 - g)
        # K0 = √(1-γ) * I
        k0 = torch.eye(2, dtype=self.tensor.real.dtype, device=self.device) * sqrt_1mg
        # Projectors
        k1 = torch.diag(torch.tensor([sqrt_g, 0.0], dtype=self.tensor.real.dtype, device=self.device))
        k2 = torch.diag(torch.tensor([0.0, sqrt_g], dtype=self.tensor.real.dtype, device=self.device))
        kraus = [k0.to(self.tensor), k1.to(self.tensor), k2.to(self.tensor)]
        new_rho = torch.zeros_like(self.tensor)
        for K in kraus:
            mats = []
            for q in range(self.num_qubits):
                mats.append(K if q == qubit else torch.eye(2, dtype=self.dtype, device=self.device))
            U = mats[0]
            for m in mats[1:]:
                U = torch.kron(U, m)
            new_rho += U @ self.tensor @ U.conj().T
        self.tensor = new_rho
        self._maybe_rescale()

    # ------------------------------------------------------------------
    # Внутреннее: авто-рескейл для mixed precision
    # ------------------------------------------------------------------
    def _maybe_rescale(self) -> None:
        if not getattr(self, "autorescale", False):
            return
        max_abs = self.tensor.abs().max()
        if max_abs > 32.0:
            self.tensor /= max_abs
        elif max_abs < 1e-4 and max_abs != 0.0:
            self.tensor /= max_abs