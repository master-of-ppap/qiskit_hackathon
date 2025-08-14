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


import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CXGate, CZGate, SwapGate, ECRGate, RZZGate, RZXGate,
    UGate, U1Gate, U2Gate, U3Gate, RXGate, RYGate, RZGate,
    SXGate, XGate, YGate, ZGate, HGate, TGate, SdgGate, SGate, TdgGate,
    PhaseGate, RGate, RYYGate, XXPlusYYGate, XXMinusYYGate
)
import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CXGate, CZGate, SwapGate, ECRGate, RZZGate, RZXGate,
    UGate, U1Gate, U2Gate, U3Gate, RXGate, RYGate, RZGate,
    SXGate, XGate, YGate, ZGate, HGate, TGate, SdgGate, SGate, TdgGate,
    PhaseGate, RGate, RYYGate, XXPlusYYGate, XXMinusYYGate
)
import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CXGate, CZGate, SwapGate, ECRGate, RZZGate, RZXGate,
    UGate, U1Gate, U2Gate, U3Gate, RXGate, RYGate, RZGate,
    SXGate, XGate, YGate, ZGate, HGate, TGate, SdgGate, SGate, TdgGate,
    PhaseGate, RGate, RYYGate, XXPlusYYGate, XXMinusYYGate
)
def random_light_circuit(num_qubits, depth, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    qc = QuantumCircuit(num_qubits)

    single_qubit_gates = [
        XGate(), YGate(), ZGate(), HGate(),
        SXGate(), SGate(), SdgGate(), TGate(), TdgGate(),
        UGate(np.pi/4, np.pi/4, np.pi/4), U1Gate(np.pi/3), U2Gate(np.pi/5, np.pi/7),
        U3Gate(np.pi/6, np.pi/8, np.pi/9), PhaseGate(np.pi/5),
        RXGate(np.pi/4), RYGate(np.pi/4), RZGate(np.pi/4),
        RGate(np.pi/4, np.pi/6)
    ]
    single_qubit_param = [
        lambda: RXGate(2 * np.pi * random.random()),
        lambda: RYGate(2 * np.pi * random.random()),
        lambda: RZGate(2 * np.pi * random.random()),
        lambda: UGate(2 * np.pi * random.random(),
                      2 * np.pi * random.random(),
                      2 * np.pi * random.random()),
        lambda: PhaseGate(2 * np.pi * random.random())
    ]
    multi_qubit_gates = [
        CXGate(), CZGate(), SwapGate(), ECRGate(),
        lambda: RZZGate(2 * np.pi * random.random()),
        lambda: RZXGate(2 * np.pi * random.random()),
        lambda: RYYGate(2 * np.pi * random.random()),
        lambda: XXPlusYYGate(2 * np.pi * random.random()),
        lambda: XXMinusYYGate(2 * np.pi * random.random())
    ]

    for _ in range(depth):
        used_qubits = set()
        q = 0
        while q < num_qubits:
            # Try a two-qubit gate if possible
            if q < num_qubits - 1 and random.random() < 0.5 and q not in used_qubits and (q + 1) not in used_qubits:
                gate_choice = random.choice(multi_qubit_gates)
                gate = gate_choice() if callable(gate_choice) else gate_choice
                qc.append(gate, [q, q + 1])
                used_qubits.add(q)
                used_qubits.add(q + 1)
                q += 2
            else:
                # Fill with single-qubit gate
                gate = (random.choice(single_qubit_gates)
                        if random.random() < 0.5
                        else random.choice(single_qubit_param)())
                qc.append(gate, [q])
                used_qubits.add(q)
                q += 1

    return qc


if __name__ == "__main__":
    # 

    random_circuit = random_light_circuit(num_qubits=12, depth=3, seed=42)
    import matplotlib.pyplot as plt
    random_circuit.draw('mpl', idle_wires=True, fold=60, scale=0.5)
    plt.show()