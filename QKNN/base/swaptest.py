import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# def norm(v):
#   v = np.array(v, dtype=float)
#   norm_val = np.linalg.norm(v)
#   if norm_val == 0 or np.isnan(norm_val):
#     raise ValueError("cannot normalize vector zero")
  
#   v = v / norm_val

#   d = len(v)
#   n_qubits = int(np.ceil(np.log2(d)))
#   dim = 2 ** n_qubits

#   padded = np.zeros(dim)
#   padded[:d] = v

#   return padded, n_qubits

def norm(v):
  v = np.array(v, dtype=float)

  d = len(v)
  n_qubits = int(np.ceil(np.log2(d)))
  dim = 2 ** n_qubits

  padded = np.zeros(dim)
  padded[:d] = v

  norm_val = np.linalg.norm(padded)
  if norm_val == 0 or np.isnan(norm_val):
     raise ValueError("cannot normalize vector zero")
  padded = padded / norm_val
  return padded, n_qubits

def encode_vector_to_state(v, qc, qubits):
  qc.initialize(v, qubits)

def swap_test(v1, v2, shots=4096):
  v1, nq = norm(v1)
  v2, _ = norm(v2)

  qc = QuantumCircuit(1 + 2 * nq, 1)

  ancilla = 0
  reg1 = list(range(1, 1 + nq))
  reg2 = list(range(1 + nq, 1 + 2 * nq))

  encode_vector_to_state(v1, qc, reg1)
  encode_vector_to_state(v2, qc, reg2)

  qc.h(ancilla)
  for i in range(nq):
    qc.cswap(ancilla, reg1[i], reg2[i])
  qc.h(ancilla)
  
  qc.measure(ancilla, 0)

  simulator = AerSimulator()
  job = simulator.run(qc, shots=shots)
  result = job.result()

  counts = result.get_counts()

  P0 = counts.get('0', 0) / shots
  P1 = counts.get('1', 0) / shots

  fidelity = P0 - P1
  # fidelity = 2 * P0 - 1

  dist = np.sqrt(1 - fidelity)

  return P0, P1, fidelity, dist
