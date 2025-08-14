
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

backend = FakeBrisbane()

## DO NOT CHANGE THE SEED NUMBER
seed = 103

## Create circuit

EPSILON = 0.30  # 10% error tolerance
DELTA = 0.10   # 5% chance of failure (95% confidence)

num_qubits = 10
depth = 10
qc1 = random_circuit(num_qubits,depth,measure=False, seed=seed)

# print("RANDOM CIRCUIT GENERATED:")
print(qc1)
# qc1.draw('mpl', idle_wires=True, fold=60, scale=0.5)

pm_lv3 = generate_preset_pass_manager(basis_gates=backend.configuration().basis_gates, optimization_level=3, seed_transpiler=seed, approximation_degree=1)

# Real circuit layout on the device
# pm_lv3 = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=seed, approximation_degree=1)

tr_random = pm_lv3.run(qc1)
# # tr_random = qc2
phase = tr_random.global_phase

tr_random.draw('mpl', idle_wires=False, fold=60, scale=0.5)
# print("TRANSPILED CIRCUIT: ")
print(tr_random)
# plt.show()

# sv_transpiled = Statevector.from_instruction(tr_random)
# sv_qc1 = Statevector.from_instruction(qc1)
# print("Transpiled circuit matches reference statevector:", sv_transpiled.equiv(sv_qc1))

# sv_transpiled = np.array(sv_transpiled)

# rotation = cos(phase) + 1j * sin(phase)
# rotated = sv_transpiled * rotation

# print(rotated)
# print(np.array(sv_qc1))

# # qc1.global_phase = -phase  # Set the global phase of the original circuit

# # Print gllobla phase of both circuits
# print("Global phase of original circuit:", qc1.global_phase)
# print("Global phase of transpiled circuit:", tr_random.global_phase)

from erft_quantum import erft
erft_result = erft(qc1, tr_random, epsilon=EPSILON, delta=DELTA, seed=seed)