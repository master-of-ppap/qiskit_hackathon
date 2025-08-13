# Qiskit State Characterization Pipeline
# This script implements and compares three methods for characterizing a quantum state
# produced by a circuit, as requested:
# 1. Direct Fidelity Estimation (DFE): For sample-efficient fidelity estimation to a known pure state.
# 2. Classical Shadows: For reconstructing an approximate state and other observables.
# 3. Full State Tomography with MLE: As a baseline for small qubit systems.
#
# The script is designed to be run with the latest versions of Qiskit.
# NOTE: qiskit-ignis is deprecated. This version uses qiskit-experiments for mitigation.
# pip install qiskit qiskit-aer qiskit-experiments

import numpy as np
import itertools
from collections import defaultdict

# Qiskit Core
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Pauli, pauli_basis, state_fidelity, DensityMatrix
# from qiskit.primitives import Sampler

# Qiskit Aer for simulation
from qiskit_aer import AerSimulator

# Qiskit Experiments for Tomography and Mitigation
from qiskit_experiments.library import StateTomography
from qiskit_experiments.library.characterization import LocalReadoutError
# from qiskit.result.mitigation.local_readout_mitigator import LocalReadoutMitigator


# ==============================================================================
# SECTION 0: HELPER FUNCTIONS
# ==============================================================================

def get_hermitian_pauli_basis(num_qubits):
    """
    Generates the standard Hermitian Pauli basis for a given number of qubits.
    This creates 4^n Pauli strings without any phase factors.

    Args:
        num_qubits (int): The number of qubits.
        
    Returns:
        list[Pauli]: A list of Pauli objects for each basis string.
    """
    pauli_strings = [''.join(p) for p in itertools.product('IXYZ', repeat=num_qubits)]
    return [Pauli(s) for s in pauli_strings]

def bootstrap_resample(data, num_resamples=1000):
    """
    Performs bootstrap resampling on a dataset to estimate confidence intervals.

    Args:
        data (np.ndarray): The dataset to resample from.
        num_resamples (int): The number of bootstrap samples to generate.

    Yields:
        np.ndarray: A resampled dataset of the same size as the original.
    """
    n_samples = len(data)
    # Generate random indices with replacement
    resample_indices = np.random.randint(0, n_samples, (num_resamples, n_samples))
    for indices in resample_indices:
        yield data[indices]


# ==============================================================================
# SECTION A: CALIBRATION & MEASUREMENT ERROR MITIGATION (using Qiskit Experiments)
# ==============================================================================

def get_readout_mitigator(num_qubits, backend, shots=8192):
    """
    Generates and runs calibration circuits to create a measurement error mitigator
    using the modern qiskit-experiments framework.

    Args:
        num_qubits (int): The number of qubits in the system.
        backend (qiskit.providers.Backend): The backend to run the calibration on.
        shots (int): The number of shots for each calibration circuit.

    Returns:
        LocalReadoutMitigator: A mitigator object that can be used to correct results.
    """
    print("--- Starting Measurement Error Mitigation (using qiskit-experiments) ---")
    
    # 1. Create a LocalReadoutError experiment.
    qubits = list(range(num_qubits))
    readout_error_exp = LocalReadoutError(qubits)
    
    # 2. Run the experiment on the backend.
    print(f"Running readout error calibration circuits on {backend.name}...")
    exp_data = readout_error_exp.run(backend, shots=shots)
    print("Calibration run complete.")
    
    # 3. Get the mitigator object from the experiment results.
    mitigator = exp_data.analysis_results(0).value
    
    print("Mitigator prepared.")
    print("Assignment matrices (confusion matrices for each qubit):")
    for i, matrix in enumerate(mitigator.assignment_matrices()):
        print(f"  Qubit {i}:\n{np.round(matrix, 2)}")
    print("Measurement error mitigation setup is complete.\n")
    
    return mitigator


# ==============================================================================
# SECTION B: TRACK 1 - DIRECT FIDELITY ESTIMATION (DFE)
# ==============================================================================

def run_dfe(target_circuit, target_state, backend, meas_mitigator=None, k_paulis=100, shots_per_pauli=100):
    """
    Performs Direct Fidelity Estimation (DFE) for a given target state.
    This version is rewritten with the correct estimator and extensive debugging prints.

    Args:
        target_circuit (QuantumCircuit): The circuit that prepares the state to be tested.
        target_state (Statevector): The ideal target statevector |psi>.
        backend (qiskit.providers.Backend): The backend to run on.
        meas_mitigator (LocalReadoutMitigator, optional): The measurement error mitigator.
        k_paulis (int): The number of Pauli operators to sample for the estimation.
        shots_per_pauli (int): The number of shots for each Pauli measurement.

    Returns:
        tuple: A tuple containing (estimated_fidelity, standard_error).
    """
    print("--- Starting Direct Fidelity Estimation (DFE) with Debugging ---")
    num_qubits = target_circuit.num_qubits
    d = 2**num_qubits
    print(f"[DFE DEBUG] Dimension d = 2^{num_qubits} = {d}")
    # EXPECTED for 3 qubits: d = 8

    # === Step 1: Pre-computation ===
    pauli_basis_set = get_hermitian_pauli_basis(num_qubits)
    s_k_array = np.array([target_state.expectation_value(p).real for p in pauli_basis_set])
    
    # === Step 2: Define Importance Sampling Distribution (using s_k^2) ===
    s_k_squared = s_k_array**2
    sum_s_k_squared = np.sum(s_k_squared)
    p_k_array = s_k_squared / sum_s_k_squared

    print(f"[DFE DEBUG] Sum of s_k^2 = {sum_s_k_squared:.4f}")
    # EXPECTED for 3-qubit GHZ: There are 8 Paulis with s_k = +/-1. So sum should be 8.0

    # === Step 3: Calculate the Estimator Prefactor ===
    prefactor = sum_s_k_squared / d
    print(f"[DFE DEBUG] Estimator prefactor (Sum(s_k^2) / d) = {prefactor:.4f}")
    # EXPECTED for 3-qubit GHZ: 8.0 / 8 = 1.0

    # === Step 4: Sample Paulis and Run Experiments ===
    sampled_indices = np.random.choice(len(pauli_basis_set), size=k_paulis, p=p_k_array)
    print(f"[DFE DEBUG] Sampled {k_paulis} Pauli operators.")

    dfe_circuits = []
    for idx in sampled_indices:
        pauli = pauli_basis_set[idx]
        meas_circ = target_circuit.copy()
        pauli_label = pauli.to_label()
        for q_idx, pauli_char in enumerate(pauli_label):
            qubit = num_qubits - 1 - q_idx
            if pauli_char == 'X': meas_circ.h(qubit)
            elif pauli_char == 'Y': meas_circ.sdg(qubit); meas_circ.h(qubit)
        meas_circ.measure_all()
        dfe_circuits.append(meas_circ)
        
    transpiled_dfe_circuits = transpile(dfe_circuits, backend)
    job = backend.run(transpiled_dfe_circuits, shots=shots_per_pauli)
    results = job.result()
    
    if meas_mitigator:
        counts_list = meas_mitigator.mitigated_counts(results.get_counts())
    else:
        counts_list = results.get_counts()

    # === Step 5: Calculate the Summand of the Estimator ===
    terms_in_sum = []
    print("\n[DFE DEBUG] Analyzing first 5 samples...")
    for i in range(k_paulis):
        idx = sampled_indices[i]
        pauli_j = pauli_basis_set[idx]
        pauli_label = pauli_j.to_label()
        s_j = s_k_array[idx]
        counts = counts_list[i]
        
        # Calculate experimental expectation value
        exp_val_j = 0
        total_shots = sum(counts.values())
        if total_shots > 0:
            for outcome, num_shots in counts.items():
                # BUG FIX: The parity must be calculated based on the specific
                # Pauli that was measured, not just all qubits.
                # We sum the bits only for the non-Identity positions.
                parity = 0
                for q_idx, pauli_char in enumerate(pauli_label):
                    if pauli_char != 'I':
                        # Qiskit bitstring is reversed, so outcome[0] is for qubit n-1
                        bit_position = q_idx
                        parity += int(outcome[bit_position])
                parity %= 2
                exp_val_j += ((-1)**parity) * num_shots
            exp_val_j /= total_shots
        
        if not np.isclose(s_j, 0):
            term = exp_val_j / s_j
            terms_in_sum.append(term)

        if i < 5:
            print(f"  Sample {i+1}:")
            print(f"    Pauli = {pauli_label}, s_j = {s_j:.4f}")
            print(f"    Measured <P_j> = {exp_val_j:.4f}")
            print(f"    Term (<P_j>/s_j) = {term:.4f}")
            # EXPECTED: For an ideal run, <P_j> should be very close to s_j,
            # so the term should be very close to 1.0. This should now work for ALL Paulis.

    # === Step 6: Calculate Final Fidelity ===
    mean_of_terms = np.mean(terms_in_sum) if terms_in_sum else 0
    std_dev_of_terms = np.std(terms_in_sum) if terms_in_sum else 0

    print(f"\n[DFE DEBUG] Mean of terms (<P_j>/s_j) = {mean_of_terms:.4f}")
    # EXPECTED: Should be close to 1.0

    estimated_fidelity = prefactor * mean_of_terms
    standard_error = prefactor * (std_dev_of_terms / np.sqrt(len(terms_in_sum))) if terms_in_sum else 0
    
    print(f"[DFE DEBUG] Final Fidelity (Prefactor * Mean) = {estimated_fidelity:.4f}")
    print("DFE complete.")
    return estimated_fidelity, standard_error


# ==============================================================================
# SECTION C: TRACK 2 - CLASSICAL SHADOWS
# ==============================================================================

def run_classical_shadows(target_circuit, backend, num_shadows=2000):
    """
    Generates a classical shadow representation of a quantum state.
    """
    print("--- Starting Classical Shadows ---")
    num_qubits = target_circuit.num_qubits

    # 1. Generate random Clifford unitaries for measurement.
    shadow_circuits = []
    unitary_indices = np.random.randint(0, 3, size=(num_shadows, num_qubits)) # 0:Z, 1:X, 2:Y
    
    for i in range(num_shadows):
        shadow_circ = target_circuit.copy(f"shadow_shot_{i}")
        for q in range(num_qubits):
            unitary_idx = unitary_indices[i, q]
            if unitary_idx == 1: shadow_circ.h(q)
            elif unitary_idx == 2: shadow_circ.sdg(q); shadow_circ.h(q)
        shadow_circ.measure_all()
        shadow_circuits.append(shadow_circ)
        
    print(f"Generated {num_shadows} circuits for shadow tomography (1 shot each).")

    # 2. Run the circuits.
    transpiled_shadow_circuits = transpile(shadow_circuits, backend)
    job = backend.run(transpiled_shadow_circuits, shots=1)
    results = job.result()

    # 3. Process the results to build the shadow.
    rho_0, rho_1 = np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])
    I_1q = np.eye(2)
    pauli_inverses = {
        (0, '0'): 3 * rho_0 - I_1q, (0, '1'): 3 * rho_1 - I_1q,
        (1, '0'): 3 * Statevector.from_label('+').to_operator().data - I_1q,
        (1, '1'): 3 * Statevector.from_label('-').to_operator().data - I_1q,
        (2, '0'): 3 * Statevector.from_label('r').to_operator().data - I_1q,
        (2, '1'): 3 * Statevector.from_label('l').to_operator().data - I_1q,
    }

    shadow_snapshots = []
    for i in range(num_shadows):
        outcome_str = list(results.get_counts(i).keys())[0]
        full_snapshot = np.array([[1.0]])
        for q in range(num_qubits):
            unitary_idx = unitary_indices[i, q]
            bit = outcome_str[num_qubits - 1 - q]
            single_qubit_snapshot = pauli_inverses[(unitary_idx, bit)]
            full_snapshot = np.kron(full_snapshot, single_qubit_snapshot)
        shadow_snapshots.append(full_snapshot)

    # 4. Aggregate using Median-of-Means.
    print("Aggregating snapshots using Median-of-Means...")
    num_batches = int(np.sqrt(num_shadows))
    snapshots_per_batch = num_shadows // num_batches
    batch_means = [np.mean(shadow_snapshots[i*snapshots_per_batch:(i+1)*snapshots_per_batch], axis=0) for i in range(num_batches)]
    rho_hat_shadow = np.median(batch_means, axis=0)
    
    # 5. Project to physical state and extract statevector.
    eigvals, eigvecs = np.linalg.eigh(rho_hat_shadow)
    top_eigenvector = eigvecs[:, -1]
    
    print("Classical Shadows processing complete.")
    return Statevector(top_eigenvector), np.array(shadow_snapshots)


# ==============================================================================
# SECTION D: (SMALL N) ALTERNATIVE - MLE TOMOGRAPHY
# ==============================================================================

def run_mle_tomography(target_circuit, backend, meas_mitigator=None, shots_per_setting=4096):
    """
    Performs full state tomography using Qiskit Experiments.
    """
    num_qubits = target_circuit.num_qubits
    if num_qubits > 4:
        print("--- MLE Tomography Warning: Skipping for >4 qubits. ---")
        return None
        
    print("--- Starting Full State Tomography (MLE) ---")
    
    # 1. Set up the StateTomography experiment.
    tomo_exp = StateTomography(target_circuit)
    
    # 2. Run the experiment.
    print(f"Running {len(tomo_exp.circuits())} tomography circuits on {backend.name}...")
    
    if meas_mitigator:
        print("Applying measurement error mitigation to tomography.")
        tomo_data = tomo_exp.run(backend, shots=shots_per_setting, measurement_mitigation=meas_mitigator)
    else:
        print("Skipping measurement error mitigation for tomography.")
        tomo_data = tomo_exp.run(backend, shots=shots_per_setting)
    
    print("Tomography run complete. Starting MLE fit...")
    
    # 3. Reconstruct the state using Maximum Likelihood Estimation (MLE).
    reconstructed_state = tomo_data.analysis_results("state").value
    
    print("MLE Tomography complete.")
    return reconstructed_state


# ==============================================================================
# SECTION E: MAIN EXECUTION AND COMPARISON
# ==============================================================================

if __name__ == '__main__':
    # --- Setup Parameters ---
    NUM_QUBITS = 3
    backend = AerSimulator()

    # --- Define Target Circuit and State ---
    print(f"Target: {NUM_QUBITS}-qubit GHZ state = (|000> + |111>)/sqrt(2)")
    target_circuit = QuantumCircuit(NUM_QUBITS)
    target_circuit.h(0)
    target_circuit.cx(0, 1)
    target_circuit.cx(0, 2)
    target_state = Statevector(target_circuit)
    
    print("\n" + "="*50)
    print("Starting Quantum State Characterization Pipeline")
    print("="*50 + "\n")

    # --- A. Run Measurement Mitigation ---
    # meas_mitigator = get_readout_mitigator(NUM_QUBITS, backend)
    meas_mitigator = None
    print("--- Skipping Measurement Error Mitigation for this ideal simulation demo ---\n")

    # --- B. Run DFE ---
    dfe_fidelity, dfe_error = run_dfe(
        target_circuit, 
        target_state, 
        backend, 
        meas_mitigator=meas_mitigator,
        k_paulis=500,
        shots_per_pauli=500
    )
    print(f"\nDFE Result: Fidelity = {dfe_fidelity:.4f} ± {dfe_error:.4f}\n")
    
    # --- C. Run Classical Shadows ---
    shadow_statevector, shadow_data = run_classical_shadows(
        target_circuit,
        backend,
        num_shadows=10000
    )
    shadow_fidelity = state_fidelity(target_state, shadow_statevector)
    
    bootstrap_fidelities = []
    for resampled_data in bootstrap_resample(shadow_data, num_resamples=100):
        num_batches = int(np.sqrt(len(resampled_data)))
        if num_batches == 0: continue
        snapshots_per_batch = len(resampled_data) // num_batches
        batch_means = [np.mean(resampled_data[i*snapshots_per_batch:(i+1)*snapshots_per_batch], axis=0) for i in range(num_batches)]
        rho_hat_resampled = np.median(batch_means, axis=0)
        _, eigvecs = np.linalg.eigh(rho_hat_resampled)
        vec_resampled = Statevector(eigvecs[:, -1])
        bootstrap_fidelities.append(state_fidelity(target_state, vec_resampled))
        
    shadow_ci = np.percentile(bootstrap_fidelities, [2.5, 97.5]) if bootstrap_fidelities else [0,0]
    shadow_error = (shadow_ci[1] - shadow_ci[0]) / 2

    print(f"\nClassical Shadows Result: Fidelity = {shadow_fidelity:.4f}")
    print(f"Reconstructed Statevector (top eigenvector): {np.round(shadow_statevector.data, 2)}")
    print(f"95% Confidence Interval (Bootstrap): [{shadow_ci[0]:.4f}, {shadow_ci[1]:.4f}]\n")

    # --- D. Run MLE Tomography ---
    mle_density_matrix = run_mle_tomography(
        target_circuit,
        backend,
        meas_mitigator=meas_mitigator,
        shots_per_setting=2048
    )
    
    if mle_density_matrix:
        mle_fidelity = state_fidelity(target_state, mle_density_matrix)
        print(f"\nMLE Tomography Result: Fidelity = {mle_fidelity:.4f}")
        print(f"Reconstructed Density Matrix:\n{np.round(mle_density_matrix.data, 2)}\n")
    else:
        mle_fidelity = "Skipped"

    # --- E. Final Summary ---
    print("\n" + "="*50)
    print("            Pipeline Summary")
    print("="*50)
    print(f"Target State: {NUM_QUBITS}-qubit GHZ State")
    print(f"Backend: {backend.name}")
    print("-" * 50)
    print(f"Direct Fidelity Estimation (DFE):   F = {dfe_fidelity:.4f} ± {dfe_error:.4f}")
    print(f"Classical Shadows + Rank-1 Proj:  F = {shadow_fidelity:.4f} (95% CI width ≈ {2*shadow_error:.4f})")
    if mle_density_matrix:
        print(f"Full Tomography (MLE):            F = {mle_fidelity:.4f}")
    print("="*50)

