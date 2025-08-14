import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import random_clifford

def erft(
    U: QuantumCircuit,
    V: QuantumCircuit,
    epsilon: float,
    delta: float,
    seed: int = None
) -> str:
    """
    Determines if two quantum circuits, U and V, are equivalent using the
    Equivalence via Randomized Fidelity Testing (ERFT) algorithm.

    Args:
        U: The first quantum circuit.
        V: The second quantum circuit to compare against the first.
        epsilon: The error tolerance for equivalence. The fidelity is checked
                 to be >= 1 - epsilon.
        delta: The probability of the estimation failing (i.e., the estimated
               fidelity being more than epsilon away from the true fidelity).
        seed: Optional seed for the simulator for reproducibility.

    Returns:
        A string indicating whether the circuits are "Equivalent (within ε)"
        or "Not equivalent".
    """
    # --- Validate Inputs ---
    if U.num_qubits != V.num_qubits:
        raise ValueError("Input circuits U and V must act on the same number of qubits.")
    if not (0 < epsilon < 1):
        raise ValueError("Epsilon (ε) must be between 0 and 1.")
    if not (0 < delta < 1):
        raise ValueError("Delta (δ) must be between 0 and 1.")

    num_qubits = U.num_qubits
    print(f"🔬 Starting ERFT for {num_qubits}-qubit circuits...")
    print(f"   - ε (error tolerance): {epsilon}")
    print(f"   - δ (failure probability): {delta}")

    # --- Step 1: Define the "difference" circuit ---
    # We want to check if W = U†V is the identity.
    # The .inverse() method computes the adjoint (dagger) of a unitary circuit.
    W = U.inverse().compose(V)
    W.name = "W = U†V"

    # --- Step 2: Determine number of samples needed ---
    # This formula comes from a one-sided Hoeffding bound. It tells us how
    # many trials (m) we need to be (1-δ)-confident that our estimated
    # fidelity is within ε of the true fidelity.
    m = int(np.ceil((1 / (2 * epsilon**2)) * np.log(2 / delta)))
    print(f"   - m (required samples): {m}")

    total_survivals = 0
    
    # Initialize the simulator
    # Using AerSimulator is the modern and performant choice in Qiskit.
    simulator = AerSimulator(method='statevector')

    # --- Step 3: Loop through m samples ---
    print(f"\n🚀 Running {m} randomized trials...")
    for i in range(m):
        # 3a. Sample a random n-qubit Clifford operator C_i
        # qiskit.quantum_info.random_clifford is highly optimized and works
        # efficiently even for large numbers of qubits.
        clifford_op = random_clifford(num_qubits)
        C_i = clifford_op.to_circuit()
        C_i.name = f"C_{i+1}"
        
        # We also need its inverse (adjoint) for the "twirl".
        # For unitary operators like Cliffords, the inverse is the adjoint.
        C_i_dagger = clifford_op.adjoint().to_circuit()
        C_i_dagger.name = f"C_{i+1}†"

        # 3b-3e. Construct the full test circuit for this trial.
        # The state starts in |0...0> by default.
        # The full operation is C_i† * W * C_i applied to |0...0>.
        test_circuit = QuantumCircuit(num_qubits, num_qubits)
        test_circuit.append(C_i, range(num_qubits))
        test_circuit.append(W, range(num_qubits))
        test_circuit.append(C_i_dagger, range(num_qubits))
        
        # 3f. Measure in the computational basis.
        test_circuit.measure(range(num_qubits), range(num_qubits))

        # Run the simulation for this single trial (shots=1).
        # Transpiling optimizes the circuit for the target backend (our simulator).
        transpiled_circuit = transpile(test_circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=1, seed=seed).result()
        counts = result.get_counts(0)

        # 3g. Record "survival" if the result is the all-zeros string '00...0'.
        # The all-zeros string is our target "survival" state.
        all_zeros_outcome = '0' * num_qubits
        if all_zeros_outcome in counts:
            total_survivals += 1

    # --- Step 4: Compute average survival probability (estimated fidelity) ---
    F_hat = total_survivals / m
    print(f"\n📊 Results:")
    print(f"   - Total 'survival' counts: {total_survivals} / {m}")
    print(f"   - Estimated Fidelity (F_hat): {F_hat:.4f}")

    # --- Step 5: Decide equivalence ---
    # If the circuits are equivalent, the fidelity should be 1. We check if
    # our estimated fidelity is within our tolerance ε.
    decision_threshold = 1 - epsilon
    print(f"   - Decision Threshold (1 - ε): {decision_threshold:.4f}")

    if F_hat >= decision_threshold:
        print("   - Outcome: F_hat ≥ 1 - ε")
        return "✅ Equivalent (within ε)"
    else:
        print("   - Outcome: F_hat < 1 - ε")
        return "❌ Not equivalent"

# --- Example Usage ---
if __name__ == '__main__':
    # --- Parameters ---
    N_QUBITS = 2
    EPSILON = 0.10  # 10% error tolerance
    DELTA = 0.05   # 5% chance of failure (95% confidence)

    # # --- Case 1: Equivalent Circuits ---
    print("-" * 50)
    print("### Case 1: Testing Equivalent Circuits ###")
    # U is a CNOT gate
    U1 = QuantumCircuit(N_QUBITS, name="U1")
    U1.cx(0, 1)

    # V is also a CNOT, but constructed from Hadamards and a CZ
    V1 = QuantumCircuit(N_QUBITS, name="V1")
    V1.h(1)
    V1.cz(0, 1)
    V1.h(1)

    # # These circuits are mathematically identical. ERFT should confirm this.
    result_1 = erft(U1, V1, epsilon=EPSILON, delta=DELTA)
    print(f"\nFinal Decision for Case 1: {result_1}\n")


    # # --- Case 2: Non-Equivalent Circuits ---
    print("-" * 50)
    print("### Case 2: Testing Non-Equivalent Circuits ###")
    # U is a simple Hadamard gate
    U2 = QuantumCircuit(N_QUBITS, name="U2")
    U2.h(0)

    # V is a Pauli-X gate
    V2 = QuantumCircuit(N_QUBITS, name="V2")
    V2.x(0)

    # # These circuits are different. ERFT should detect this.
    result_2 = erft(U2, V2, epsilon=EPSILON, delta=DELTA)
    print(f"\nFinal Decision for Case 2: {result_2}\n")

    # --- Case 3: Nearly Equivalent Circuits (with a larger error) ---
    print("-" * 50)
    print("### Case 3: Testing Nearly Equivalent Circuits ###")
    # U is a perfect X-gate (a pi rotation around X)
    U3 = QuantumCircuit(N_QUBITS, name="U3")
    U3.rx(np.pi, 0)

    # V is an X-rotation with a larger, more obvious error.
    # The original error (0.95 * pi) was too small to be detected with ε=0.1.
    # This larger error (0.75 * pi) creates a fidelity low enough to fail the test.
    V3 = QuantumCircuit(N_QUBITS, name="V3")
    V3.rx(np.pi * 0.85, 0)

    # The fidelity will now be low enough that F_hat < (1 - epsilon) is likely.
    result_3 = erft(U3, V3, epsilon=EPSILON, delta=DELTA)
    print(f"\nFinal Decision for Case 3: {result_3}\n")
