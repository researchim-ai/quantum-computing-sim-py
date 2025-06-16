# quantum-computing-sim-py
Quantum computing simulator

Минимальный квантовый симулятор на PyTorch.

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
