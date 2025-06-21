# quantum-computing-sim-py
Quantum computing simulator

Минимальный квантовый симулятор на PyTorch.

## Возможности

* **State-vector** и **Density-matrix** backend'ы на `torch.Tensor` (autograd-ready).
* Поддержка однокубитных, двухкубитных (CX, CZ, SWAP) и универсального `U3` гейта.
* Kraus-шумовые каналы: depolarizing, amplitude-damp.
* **GPU-ускорение** через Triton-kernels для single, CX, CZ, SWAP.
* CLI `qsimx run` с QASM-парсером и опциями `--backend`, `--noise`.
* RL-среда `CircuitDesignEnv` (Gymnasium-совместимая).
* **Бенчмарк** скрипт `bench/bench_speed.py` (CPU vs CUDA, учёт памяти).
* Полное покрытие PyTest + Sphinx-документация.

## Установка

```bash
pip install -r requirements.txt  # или poetry install
```

## Быстрый пример

```python
from qsimx import QuantumCircuit

circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)
state = circ.simulate(device="cuda")
print(state.abs() ** 2)  # [0.5, 0, 0, 0.5]
```

Запуск из CLI:

```bash
qsimx run "H0,CX0-1" --device cuda
```

## Benchmark

Скрипт `bench/bench_speed.py` сравнивает CPU-einsum и GPU-Triton.

```bash
# 30 кубитов, глубина 512, запускаем на GPU0 и CPU
python bench/bench_speed.py -n 30 -d 512 \
       --devices cuda:0,cpu --dtype c64
```

> На небольших схемах и глубинах GPU-вариант может быть медленнее из-за оверхеда запуска ядер, поэтому в CI-тестах проверяется ускорение на глубине 256 и выше.

```python
import torch
from qsimx import QuantumCircuit

# создаём схему Белла
circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)
state = circ.simulate(device="cpu")  # или "cuda"

print(state)            # амплитуды
print(state.abs() ** 2) # вероятности
```
