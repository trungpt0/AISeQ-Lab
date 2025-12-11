import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def norm(vect):
    vect = np.array(vect, dtype=float)
    norm_val = np.linalg.norm(vect)
    if norm_val == 0 or np.isnan(norm_val):
        raise ValueError("Cannot normalize zero vector")
    return vect / norm_val


def encode_vector_to_state(vect, qc, qubits):
    qc.initialize(vect, qubits)


def swap_test(vect1, vect2, shots=4096):

    vect1 = norm(vect1)
    vect2 = norm(vect2)

    qc = QuantumCircuit(3, 1)

    qc.h(0)

    encode_vector_to_state(vect1, qc, [1])
    encode_vector_to_state(vect2, qc, [2])

    qc.cswap(0, 1, 2)

    qc.h(0)

    qc.measure(0, 0)

    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()

    counts = result.get_counts()

    P0 = counts.get('0', 0) / shots
    P1 = counts.get('1', 0) / shots

    fidelity = P0 - P1
    dist = np.sqrt(1 - fidelity)

    return P0, P1, fidelity, dist
