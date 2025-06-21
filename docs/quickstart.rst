Быстрый старт
==============

.. code-block:: python

   >>> from qsimx import QuantumCircuit
   >>> circ = QuantumCircuit(2)
   >>> circ.h(0).cx(0, 1)
   >>> state = circ.simulate()
   >>> print(state)

Запуск из CLI:

.. code-block:: bash

   qsimx run "H0,CX0-1"

QASM файл:

.. code-block:: bash

   qsimx run bell.qasm --backend density --noise depol:0.02 