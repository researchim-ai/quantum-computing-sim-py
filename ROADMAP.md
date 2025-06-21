## План разработки продвинутого квантового симулятора

---

### Этап 0. Формализация целей

1. **Use-case-документ**: поддержку LLM-экспериментов (нам нужны автоградиенты) и RL-сред для агентов.
2. **MVP-метрики**: скорость моделирования > cuQuantum/qsim на тех же GPU, память ≤ 1.2× state-vector предела.
3. **Техстек**: C++20 backend + CUDA/HIP/OpenCL kernels, Python/JAX/PyTorch биндинги.

---

### Этап 1. Ядро симуляции

* **Архитектура плагино-модулей**:

  * *State-vector* (double / complex64). **✅ MVP на PyTorch готов (complex64, autograd, JIT-флаг).**
  * *Stabilizer* для Clifford-подобных схем. *(todo)*
  * *Tensor Network* с авто-поиском contraction path. *(todo)*
  * *Density-matrix* и *Kraus*-шум. *(в работе – следующий шаг)*
* **Высокопроизводительные ядра**:

  * Кузов CUDA + cutensor/cutensornet. *(todo)*
  * Мульти-GPU через NCCL + нарезка состояния (Schmidt-слайсинг). *(todo)*
  * CPU-fallback с AVX-512/NEON. *(todo)*

> **Ближайшие доработки Этапа 1**  
> • Mixed precision (FP16/BF16) + авто-rescale (частично: autorescale готов, нужны FP16-kernels).  
> • DensityMatrix класс + depolarizing/amplitude-damp Kraus.  
> • Оптимизация гейтов через Triton-kernels.  
> • **CUDA/Triton-kernels** для однокубитных гейтов и CX (Tensor Core, fusion).  
> • **CUDA Graphs** — кеш сценария схемы, уменьшение Python-overhead.  
> • **Мульти-GPU slicing** (NCCL All-to-All, overlap compute/comm).

---

### Этап 2. Расширенные физические модели

* **Шумовые каналы**: depolarizing, amplitude-damp, 1/f, T₁/T₂ с профилями по qubit-ID.
* **Квантовая коррекция**: готовые коды (Шор, Surface-17) + быстрый MWPM-декодер.
* **Аналоговый режим**: Trotter/Suzuki эволюция, моделирование пульсов с скрытым ODE-решателем.
* **Авто-дифференцируемые операции** (adjoint method, parameter-shift).
* **Дополнительные каналы**: amplitude-damp (γ), phase-damp, универсальный Lindblad супер-оператор.

---

### Этап 3. ML-интеграция

1. **Python high-level API**

   * drop-in объект `QuantumCircuit` → `.simulate()` возвращает `torch.Tensor`.
   * Поддержка *torch.autograd* и JAX `grad`.
2. **Gymnasium-среды**

   * *CircuitDesignEnv*: действие — добавить/изменить гейт, награда — fidelity / cost.
   * *QECEnv*: агент ищет декодер или роутинг.
3. **Datasets для LLM**

   * Экспорт «строка — описание схемы / строка — результат» + токенизаторы.
   * Скрипты генерации batched симуляций на GPU-кластере.

* **Градиенты**
  * Adjoint back-prop (чек-пойнты, O(m·2ⁿ)).
  * Parameter-shift / finite-diff fallback.
  * JAX backend через host_callback.

---

### Этап 4. Масштабирование и оптимизация

* **Mixed precision** (FP16/BF16) с авто-rescaling.
* **Динамический *circuit cutting*** для > 40 кубитов.
* **Distributed Tensor Networks** (MPI + GPU-RDMA).
* **JIT-компиляция** горячих путей через NVIDIA CUDA Graphs / AMD HIP graph.
* **Профилировщик**: встроенный трейсер (Chrome Trace).
* **Авто-backend-scheduler**: statevector / stabilizer / TN / slicing по размеру схемы и шуму.

---

### Этап 5. Инструменты и UX

* CLI `qsimx run my.qasm --backend gpu`.
* **Веб-дашборд** (FastAPI + React): визуализация схемы, live-графики использования GPU/памяти.
* **Визуализатор гейт-пульсов** (Plotly WebGL).
* Полная документация (Sphinx + Markdown-кодсниппеты).
* **CLI v2**: `--backend auto`, `--noise depol:0.01`, вывод Chrome-Trace.
* **Chrome Trace** профилировщик (`with qsimx.trace()`).

---

### Этап 6. Кейсы LLM + RL

* **Пример-репозитории**:

  * RL-агент (PPO) оптимизирует вариационную схему VQE.
  * LLM-агент рассуждает о комбинациях гейтов, вызывает симулятор через LangChain-tool.
* **Пайплайны**: Docker/Conda, Helm-чарты для k8s-кластеров, SLURM-скрипты.

---

### Этап 7. Валидация и комьюнити-выход

1. **Научная верификация**: сравнение с экспериментами IBM & IonQ, публикация arXiv-препринта.
2. **Бенчмарк-отчёт**: таблица *speed-up vs qubits* против cuQuantum vX.Y, qsim vZ.
3. **Open-source релиз** (Apache-2.0), гайдлайн для контрибьюторов, RFC-процесс.
4. **Property-based тесты (Hypothesis)** против Qiskit Aer / Stim.

---

### Критические компетенции команды

* HPC/GPGPU разработчик (C++/CUDA/HIP).
* Квантовый алгоритмист (тензорные сети, QEC).
* ML/RL инженер (PyTorch, JAX, Gymnasium).
* DevOps + фронтенд для UX.

---

### Риски и как их смягчить

| Риск                        | Митигатор                                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| **GPU-бутылочные горлышки** | Ранний прототип на реальных схемах 30–40 qubits, профилирование, roadmap к NCCL-over-InfiniBand. |
| **Научная корректность**    | Встроенный property-based-тест против аналитических схем + cross-checks с Aer / Braket.          |
| **Сложность API**           | Поддержка Qiskit-like синтаксиса + адаптер к Cirq, подробные туториалы.                          |
