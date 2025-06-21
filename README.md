# quantum-computing-sim-py
Quantum computing simulator

Минимальный квантовый симулятор на PyTorch.

## Benchmark

Для оценки ускорения Triton-backend добавлен скрипт `bench/bench_speed.py`.

```bash
# 20 кубитов, глубина 1024
python bench/bench_speed.py -n 20 -d 1024
```

Вывод (пример, RTX 3080):
```
Qubits 20 Depth 1024
CPU-einsum : 12.34 s
CUDA-fast  : 1.95 s
```

На небольших схемах GPU может проигрывать из-за накладных расходов, поэтому в тестах проверяется как минимум двукратное ускорение на более глубокой цепочке.

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
