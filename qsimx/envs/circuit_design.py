from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from ..circuit import QuantumCircuit
from ..statevector import StateVector

__all__ = ["CircuitDesignEnv"]


class CircuitDesignEnv(gym.Env):
    """Простая среда RL: агент по шагам строит схему и приближает заданное целевое состояние.

    Предполагается фиксированное число кубитов ``n_qubits`` и максимальная глубина
    ``max_depth``.  Действие — выбрать гейт из дискретного набора (H, X на каждом
    кубите + CX для каждой упорядоченной пары).  Награда — приращение в
    *fidelity* к целевому состоянию.
    """

    metadata = {"render_modes": [None]}

    def __init__(
        self,
        target_circuit: QuantumCircuit,
        *,
        max_depth: int = 20,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.n_qubits = target_circuit.num_qubits
        self.max_depth = max_depth
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or torch.complex64

        # Предварительно вычисляем целевое состояние
        self.target_state: torch.Tensor = target_circuit.simulate(dtype=self.dtype, device=self.device)
        # Нормируем (на случай численных ошибок)
        self.target_state = self.target_state / self.target_state.norm(p=2)

        # Служебные переменные эпизода
        self.statevector: StateVector | None = None
        self.step_count: int = 0
        self.prev_fid: float = 0.0

        # Генерируем таблицу действий -> (gate_name, *args)
        self._actions: List[Tuple[str, Tuple[int, ...]]] = []
        for q in range(self.n_qubits):
            self._actions.append(("h", (q,)))
            self._actions.append(("x", (q,)))
        for control in range(self.n_qubits):
            for target in range(self.n_qubits):
                if control == target:
                    continue
                self._actions.append(("cx", (control, target)))

        self.action_space = spaces.Discrete(len(self._actions))

        dim = 1 << self.n_qubits
        # Возвращаем наблюдение: вещественная и мнимая части амплитуд
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * dim,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        t = self.statevector.tensor
        obs = torch.cat([t.real, t.imag]).to(torch.float32).cpu().numpy()
        return obs

    def _compute_fidelity(self) -> float:
        psi = self.statevector.tensor
        fid = torch.abs(torch.dot(self.target_state.conj(), psi)) ** 2
        return fid.item()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.statevector = StateVector(self.n_qubits, dtype=self.dtype, device=self.device)
        self.step_count = 0
        self.prev_fid = self._compute_fidelity()
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.statevector is not None, "Call reset() before step()"
        gate_name, args = self._actions[action]
        getattr(self.statevector, gate_name)(*args)
        self.step_count += 1
        fid = self._compute_fidelity()
        reward = fid - self.prev_fid
        self.prev_fid = fid
        terminated = fid > 1 - 1e-6
        truncated = self.step_count >= self.max_depth
        info = {"fidelity": fid}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):  # pragma: no cover
        print(f"Step {self.step_count}, fidelity={self.prev_fid:.4f}")

    def close(self):  # pragma: no cover
        self.statevector = None 