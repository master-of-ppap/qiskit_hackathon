
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import *
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt
import math
from erft_quantum import erft

from math import cos, sin
from qiskit.converters import circuit_to_dag, dag_to_circuit
from collections import OrderedDict
import numpy as np

from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, qpy, qasm3, transpile

from qiskit_ibm_runtime import QiskitRuntimeService
from random_clifford import random_light_circuit

# backend = FakeBrisbane()
service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")
# backend = FakeBrisbane()
print(backend.name)

output_filepath = "output/"

from qiskit import QuantumCircuit

seed = 103

EPSILON = 0.1  # 10% error tolerance (Default 0.3)
DELTA = 0.05   # 5% chance of failure (95% confidence)

num_qubits = 12
depth = 2
# target_circuit.z(1)
# target_circuit.h(1)
# target_circuit.x(1)
qc1 = random_light_circuit(num_qubits=num_qubits, depth=depth, seed=seed)


output_name = output_filepath + str(num_qubits) + "_" + str(depth) + "_" + str(backend.name)

# print("RANDOM CIRCUIT GENERATED:")
# print(qc1)
qc1.draw('mpl', idle_wires=True, fold=60, scale=0.5)
plt.show()

basis_idea = ['id', 'rz', 'sx', 'x', 'cx', 'swap', 'rzx', 'ccx', 'u', 'reset']

pm_lv3 = generate_preset_pass_manager(basis_gates=basis_idea, optimization_level=3, seed_transpiler=seed, approximation_degree=1)

#
tr_random = pm_lv3.run(qc1)

phase = tr_random.global_phase

tr_random.draw('mpl', idle_wires=False, fold=60, scale=0.5)

from erft_quantum import erft
out_txt = output_name + "_results.txt"
erft_result = erft(qc1, tr_random, epsilon=EPSILON, delta=DELTA, seed=seed, outfile=out_txt, backend=backend)