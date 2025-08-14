from qiskit import QuantumCircuit
import random

def random_clifford_circuit(n_qubits: int, depth: int, seed: int = None) -> QuantumCircuit:
    """
    Create a random Clifford-only circuit with given depth and number of qubits.
    
    Args:
        n_qubits: Number of qubits.
        depth: Number of layers (depth) to create.
        seed: Optional random seed for reproducibility.
    
    Returns:
        QuantumCircuit: A circuit with random Clifford gates.
    """
    if seed is not None:
        random.seed(seed)
    
    # Clifford gate set: single-qubit (H, S, X, Y, Z), two-qubit (CX, CZ, SWAP)
    one_qubit_cliffords = ['h', 's', 'sdg', 'x', 'y', 'z']
    two_qubit_cliffords = ['cx', 'cz', 'swap']
    
    qc = QuantumCircuit(n_qubits)
    
    for _ in range(depth):
        # Randomly decide if layer will be single- or two-qubit gates
        if random.random() < 0.5:
            # Apply random 1-qubit Clifford to each qubit
            for q in range(n_qubits):
                gate = random.choice(one_qubit_cliffords)
                getattr(qc, gate)(q)
        else:
            # Apply random 2-qubit Clifford
            q1, q2 = random.sample(range(n_qubits), 2)
            gate = random.choice(two_qubit_cliffords)
            getattr(qc, gate)(q1, q2)
    
    return qc


# random_circuit = random_clifford_circuit(n_qubits=3, depth=5, seed=42)
