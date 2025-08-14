import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKyoto
import matplotlib.pyplot as plt

# import numpy as np

# from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# --- 1. Setup U (Ideal) and V (Transpiled) ---
num_qubits = 3
backend = FakeKyoto()
seed = 103

# U is the ideal CCZ gate
U = QuantumCircuit(num_qubits, name="U_ideal")
U.ccx(0, 1, 2)

# V is your transpiled circuit from the backend
pm_lv3 = generate_preset_pass_manager(optimization_level=3, seed_transpiler=seed, basis_gates=backend.configuration().basis_gates)


V_raw = pm_lv3.run(U)
V_raw.global_phase = 0.0  # Reset global phase for comparison
# V_raw.draw(output='mpl', idle_wires=False, fold=20, scale=0.8)
V = V_raw
# V = copy_circuit_no_empty_lines(V_raw)
# V.draw(output='mpl', idle_wires=False, fold=20)

# IMPORTANT: Reset the global phase to compare only the unitary operation
V.global_phase = 0.0

# --- 2. Create the Test Operator W = U_dagger * V ---
# If U and V are the same, W should be the Identity matrix
W = U.inverse().compose(V)
# W.name = "W_test"

# # --- 3. Build and Run a Test Circuit for Each Basis State ---
test_circuits = []
shots = 1024

print("Building 8 test circuits, one for each basis state input...")
for i in range(2**num_qubits):
    # Get the binary string for the input state (e.g., 5 -> '101')
    input_state_str = format(i, f'0{num_qubits}b')

    # Create a circuit for this specific test
    qc_test = QuantumCircuit(num_qubits, num_qubits, name=f"test_{input_state_str}")

    # a) Prepare the input state |i> from |000>
    # We apply X gates where the binary string has a '1'
    # The string is reversed because Qiskit's qubit order is q2, q1, q0
    for qubit_idx, bit in enumerate(reversed(input_state_str)):
        if bit == '1':
            qc_test.x(qubit_idx)
    qc_test.barrier(label=f"Input |{input_state_str}⟩")

    # b) Apply the test operator W
    qc_test.append(U, range(num_qubits))
    qc_test.barrier(label="Output")

    # c) "Un-prepare" the state by applying the preparation gates again
    # (Since X is its own inverse, this reverses the preparation)
    # for qubit_idx, bit in enumerate(reversed(input_state_str)):
    #     if bit == '1':
# /            qc_test.x(qubit_idx)

    # d) Measure. If W=I, the state is |000>, so we expect to measure '000'
    qc_test.measure(range(num_qubits), range(num_qubits))
    test_circuits.append(qc_test)

# --- 4. Execute and Analyze Results ---
simulator = AerSimulator()
# Transpile the test circuits for the simulator for efficiency
t_test_circuits = transpile(test_circuits, simulator)
job = simulator.run(t_test_circuits, shots=shots)
result = job.result()
counts_list = result.get_counts()

print("\n--- Test Results ---")
all_passed = True
for i, counts in enumerate(counts_list):
    input_state_str = format(i, f'0{num_qubits}b')
    # Check if we got 100% '000' results
    print(f"Test for input |{input_state_str}⟩. Result: {counts}")

if all_passed:
    print("\nConclusion: The transpiled circuit is functionally identical to CCZ.")
else:
    print("\nConclusion: The transpiled circuit has small deviations from an ideal CCZ.")

plt.show()
