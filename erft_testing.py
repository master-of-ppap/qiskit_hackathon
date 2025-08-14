
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

# backend = FakeBrisbane()
service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")
print(backend.name)

output_filepath = "output/"

from qiskit import QuantumCircuit

## DO NOT CHANGE THE SEED NUMBER
seed = 103

## Create circuit

EPSILON = 0.30  # 10% error tolerance (Default 0.3)
DELTA = 0.10   # 5% chance of failure (95% confidence)

num_qubits = 3
depth = 3
target_circuit = QuantumCircuit(num_qubits)
# target_circuit.z(1)
# target_circuit.h(1)
# target_circuit.x(1)
qc1 = random_circuit(num_qubits,depth,measure=False, seed=seed)

output_name = output_filepath + str(num_qubits) + "_" + str(depth) + "_" + str(backend.name)

# print("RANDOM CIRCUIT GENERATED:")
# print(qc1)
# qc1.draw('mpl', idle_wires=True, fold=60, scale=0.5)
qc1.draw(output="mpl", filename=(output_name + "_initQC.png"))
with open (output_name + "_initQC.qasm", 'w') as f:
    qasm3.dump(qc1, f)

pm_lv3 = generate_preset_pass_manager(basis_gates=backend.configuration().basis_gates, optimization_level=3, seed_transpiler=seed, approximation_degree=1)

# Real circuit layout on the device
# pm_lv3 = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=seed, approximation_degree=1)

tr_random = pm_lv3.run(qc1)
# # tr_random = qc2
phase = tr_random.global_phase

# tr_random.draw('mpl', idle_wires=False, fold=60, scale=0.5)
tr_random.draw(output="mpl", idle_wires=False, filename=(output_name + "_transpiledQC.png"))
with open (output_name + "transpiledQC.qasm", 'w') as f:
    qasm3.dump(tr_random, f)
# print("TRANSPILED CIRCUIT: ")
# print(tr_random)
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
out_txt = output_name + "_results.txt"
erft_result = erft(qc1, tr_random, epsilon=EPSILON, delta=DELTA, seed=seed, outfile=out_txt)