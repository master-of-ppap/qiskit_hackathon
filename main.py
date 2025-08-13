from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Example circuits
qc_ref = QuantumCircuit(2)
qc_ref.h(0)
qc_ref.cx(0, 1)
# Transpilation

qc_test = QuantumCircuit(2)
qc_test.h(0)
qc_test.cx(0, 1)
# Transpilation 2 



# Get statevectors
sv_ref = Statevector.from_instruction(qc_ref)
sv_test = Statevector.from_instruction(qc_test)

# Check equivalence (up to global phase)
print(sv_ref.equiv(sv_test))  # True/False
