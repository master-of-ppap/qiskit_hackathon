
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
from qiskit import QuantumCircuit, qpy, qasm3, transpile

from qiskit_ibm_runtime import QiskitRuntimeService
from random_clifford import *

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")
# backend = FakeBrisbane()
print(backend.name)

output_filepath = "output/"


seed = 105

EPSILON = 0.3  # 10% error tolerance (Default 0.3)
DELTA = 0.05   # 5% chance of failure (95% confidence)

Us = []
Vs = []
outnames = []

for i, num_qubits in enumerate(range(5, 50, 3)):
    if i % 5 == 0:
        print(f"\n\nGenerating circuits for {num_qubits} qubits...")
    for j in range(2):
        # num_qubits = 10
        depth = 5
        permutations = 5

        slightly_different = j == 1

        two_slightly_different_circuits_list = two_slightly_different_circuits(num_qubits, depth, seed=seed, permutations=permutations)

        qc1 = two_slightly_different_circuits_list[0]
        qc2 = two_slightly_different_circuits_list[1]


        output_name = output_filepath + str(num_qubits) + "_" + str(depth) + "_" + str(backend.name)
        if slightly_different:
            # print("Slightly different circuits")
            output_name += "_p=" + str(permutations)
            output_name += "_slightly_different"
            

        qc1.draw('mpl', idle_wires=True, fold=60, scale=0.5, filename=output_name + "_original.png")
        plt.close()

        basis_idea = ['id', 'rz', 'sx', 'x', 'cx', 'swap', 'rzx', 'ccx', 'u', 'reset']

        if slightly_different:
            tr_random = qc2
        else:
            pm_lv3 = generate_preset_pass_manager(basis_gates=basis_idea, optimization_level=3, seed_transpiler=seed, approximation_degree=1)
            tr_random = pm_lv3.run(qc1)
        tr_random.draw('mpl', idle_wires=True, fold=60, scale=0.5, filename=output_name + "_transpiled.png")
        plt.close()
        
        if slightly_different:
            Us.append(qc1)
            Vs.append(qc2)
        else:
            Us.append(qc1)
            Vs.append(tr_random)
        outnames.append(output_name)

# tr_random.draw('mpl', idle_wires=False, fold=60, scale=0.5)

# print(f"Len of Us: {len(Us)}")
# print(f"Outnames: {outnames}")

assert len(Us) == len(Vs) == len(outnames), "Us, Vs, and outnames must have the same length."
from erft_quantum import erft
out_txt = output_name + "_results.txt"
erft_result = erft(Us, Vs, epsilon=EPSILON, delta=DELTA, seed=seed, outfile="output/results.txt", outnames=outnames, backend=backend)

# from erft_quantum_2 import erft_2

# erft_result_2 = erft_2(qc1, tr_random, epsilon=EPSILON, delta=DELTA, seed=seed, backend=backend)