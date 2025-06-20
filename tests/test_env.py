import gymnasium as gym
import numpy as np

from qsimx import CircuitDesignEnv, QuantumCircuit


def test_env_single_episode():
    target = QuantumCircuit(1)
    target.h(0)
    env = CircuitDesignEnv(target, max_depth=5)
    obs, info = env.reset()
    assert obs.shape == (2 * 2,)  # 2 qubits amplitudes (real+imag)
    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    env.close()
    assert isinstance(total_reward, float) 