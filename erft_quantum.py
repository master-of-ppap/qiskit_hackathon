import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import random_clifford
from qiskit.transpiler.passes import SabreLayout, SabreSwap
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
import mthree

from random_clifford import *


def erft_generate_circuits(U: QuantumCircuit, V:QuantumCircuit, m, backend, seed=None):

    # --- Validate Inputs ---
    if U.num_qubits != V.num_qubits:
        raise ValueError("Input circuits U and V must act on the same number of qubits.")

    num_qubits = U.num_qubits
    # --- Step 1: Define the "difference" circuit ---
    # We want to check if W = U†V is the identity.
    # The .inverse() method computes the adjoint (dagger) of a unitary circuit.

    W = U.inverse().compose(V)
    print(f"Testing on depth of {W.depth()} and size of {W.size()} gates.")
    print(f"Depth of U: {U.depth()}, Depth of V: {V.depth()}")
    print(f"Size of U: {U.size()}, Size of V: {V.size()}")

    # --- Step 2: Determine number of samples needed ---
    # This formula comes from a one-sided Hoeffding bound. It tells us how
    # many trials (m) we need to be (1-δ)-confident that our estimated
    # fidelity is within ε of the true fidelity.
    
    print(f"   - m (required samples): {m}\n")
    print("In erft_quantum.py: " + backend.name)

    circuits = []

    print(f"\n\n  Running {m} randomized trials...\n")
    for i in range(m):

        C_i = random_clifford_circuit_2(num_qubits, depth=2, seed=seed)
        C_i_dagger = C_i.inverse()

        test_circuit = QuantumCircuit(num_qubits)
        test_circuit.append(C_i, range(num_qubits))
        test_circuit.append(W, range(num_qubits))
        test_circuit.append(C_i_dagger, range(num_qubits))

        if i == 0:
            print(f"  Trial {i+1}/{m}:\n")
            print(f"Size of test circuit: {test_circuit.size()}, depth: {test_circuit.depth()}")
            print(f"Size of C_i and C_i_dagger: {C_i.size()}, depth: {C_i.depth()}")

        # test_circuit.decompose(reps=2)

        test_circuit.measure_all()

        transpiled_circuit = transpile(test_circuit, backend, optimization_level=2)

        circuits.append(transpiled_circuit)

        if i % (m // 3) == 0 or i == m - 1:
            print(f"  Trial {i+1}/{m} completed.")
            print(f"Resulting circuit: {transpiled_circuit.size()} gates, depth {transpiled_circuit.depth()}")
        
    return circuits

def erft(
    Us: QuantumCircuit,
    Vs: QuantumCircuit,
    epsilon: float,
    delta: float,
    seed: int = None,
    outfile=None,
    outnames=list[str],
    backend=None,
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

    if len(Us) != len(Vs):
        raise ValueError("Input circuits Us and Vs must have the same number of circuits.")


    all_circuits = []
    circuit_batch_lengths = []

    for i, (U, V) in enumerate(zip(Us, Vs)):

        # --- Validate Inputs ---
        if U.num_qubits != V.num_qubits:
            raise ValueError("Input circuits U and V must act on the same number of qubits.")
        if not (0 < epsilon < 1):
            raise ValueError("Epsilon (ε) must be between 0 and 1.")
        if not (0 < delta < 1):
            raise ValueError("Delta (δ) must be between 0 and 1.")

        num_qubits = U.num_qubits

        print(f"  Starting ERFT for {i}th {num_qubits}-qubit circuits ({outfile})...\n")
        print(f"   - ε (error tolerance): {epsilon}\n")
        print(f"   - δ (failure probability): {delta}\n")


        m = int(np.ceil((1 / (2 * epsilon**2)) * np.log(2 / delta)))

        circuits = erft_generate_circuits(U, V, m, backend=backend, seed=seed)
        all_circuits.extend(circuits)
        circuit_batch_lengths.append(len(circuits))

    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(range(num_qubits))

    sampler = Sampler(mode=backend)
    F_hats = []
    results = sampler.run(all_circuits, shots=1).result()
    print(f"\n\n  Total circuits run: {len(all_circuits)}\n")
    print(f"  Total trials: {m}\n")

    idx = 0
    for i, m_batch in enumerate(circuit_batch_lengths):
        batch_results = results[idx:idx + m_batch]
        total_survivals = 0
        num_qubits = Us[i].num_qubits
        
        # Assert batch_results length matches expected m_batch
        assert len(batch_results) == m_batch, f"Batch {i}: Expected {m_batch} results, got {len(batch_results)}"

        for result in batch_results:
            counts = mit.apply_correction(result.data.meas.get_counts(), range(num_qubits))
            # Assert counts is a dictionary
            assert isinstance(counts, dict), f"Batch {i}: Counts is not a dict"
            if counts.get('0' * num_qubits, 0) > 0:
                total_survivals += 1

        # Assert total_survivals is not negative and does not exceed m_batch
        assert 0 <= total_survivals <= m_batch, f"Batch {i}: Survival count out of bounds"

        F_hat = total_survivals / m_batch
        print(f"\nResults for batch {i}:")
        print(f"  Survival counts: {total_survivals} / {m_batch}")
        print(f"  Estimated Fidelity: {F_hat:.4f}")
        decision_threshold = 1 - epsilon
        print(f"  Decision Threshold: {decision_threshold:.4f}")

        # Assert F_hat is between 0 and 1
        assert 0.0 <= F_hat <= 1.0, f"Batch {i}: F_hat out of bounds"

        if F_hat >= decision_threshold:
            print("  Outcome: Equivalent")
        else:
            print("  Outcome: Not Equivalent")
        if outfile:
            with open(outfile, "a") as f:
                
                assert len(outnames) == len(Us), "Output names must match the number of circuits."

                f.write(f"{outnames[i]} - Circuit {i}: F_hat = {F_hat:.4f}\n")
        F_hats.append(F_hat)
        idx += m_batch
    return F_hats

# --- Example Usage ---
if __name__ == '__main__':
    # --- Parameters ---
    N_QUBITS = 4
    EPSILON = 0.1  # 10% error tolerance
    DELTA = 0.05   # 5% chance of failure (95% confidence)


    U1 = QuantumCircuit(N_QUBITS, name="U1")
    U1.cx(0, 1)
    U1.cx(2, 3)


    V1 = QuantumCircuit(N_QUBITS, name="V1")
    V1.h(1)
    V1.cz(0, 1)
    V1.h(1)

    V1.h(3)
    V1.cz(2, 3)
    V1.h(3)

    result_1 = erft(U1, V1, epsilon=EPSILON, delta=DELTA)
    print(f"\n\nFinal Decision for Case 1: {result_1}\n")
