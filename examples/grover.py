"""
Small Grover's search demo. For simplicity the oracle and diffusion
operators are built as explicit 2**n x 2**n unitary matrices and applied
with QuantumCircuit.custom() across ALL n qubits at once -- this is the
O(2**n * 2**n) = O(4**n) worst case of apply_gate (k = n), included
deliberately to make the point: a real Grover implementation decomposes
the oracle/diffusion into O(n) elementary 1- and 2-qubit gates (multi-
controlled-Z via ancillas/Toffolis) to stay in the O(2**n * 2**k) regime
with small constant k. That decomposition is a circuit-design exercise
independent of the simulator itself, so it's left out of this demo.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from qsim import QuantumCircuit


def oracle_matrix(n: int, marked: int) -> np.ndarray:
    dim = 1 << n
    U = np.eye(dim, dtype=np.complex128)
    U[marked, marked] = -1
    return U


def diffusion_matrix(n: int) -> np.ndarray:
    dim = 1 << n
    s = np.ones((dim, 1)) / np.sqrt(dim)
    return 2 * (s @ s.T) - np.eye(dim)


def grover_circuit(n: int, marked: int, iterations: int | None = None) -> QuantumCircuit:
    dim = 1 << n
    if iterations is None:
        iterations = max(1, round(np.pi / 4 * np.sqrt(dim)))

    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)

    oracle = oracle_matrix(n, marked)
    diffusion = diffusion_matrix(n)
    for _ in range(iterations):
        qc.custom(oracle, list(range(n)), label="Oracle")
        qc.custom(diffusion, list(range(n)), label="Diff")
    return qc


if __name__ == "__main__":
    n = 4
    target = 11  # search for |1011>
    qc = grover_circuit(n, target)
    result = qc.run(shots=2000, seed=0)
    print(f"Searching for |{format(target, f'0{n}b')}> among {1 << n} states")
    print("Counts:", dict(sorted(result.counts.items(), key=lambda kv: -kv[1])[:5]))
