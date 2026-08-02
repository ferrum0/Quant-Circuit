"""
O(4**n)-style reference gate application, independent of qsim's optimized
tensordot-based implementation. Used by both the example-based tests
(test_qsim.py) and the property-based tests (test_properties.py) to check
correctness. Not itself a test module (no test_ prefix -> not collected).
"""
import numpy as np


def dense_apply(state: np.ndarray, matrix: np.ndarray, qubits, n) -> np.ndarray:
    """Build the full 2**n x 2**n operator by embedding `matrix` into the
    full space via explicit basis-state bit bookkeeping, then multiply.
    This is deliberately independent of the reshape/tensordot machinery in
    qsim.state.QuantumState.apply_gate."""
    k = len(qubits)
    dim = 1 << n
    full = np.zeros((dim, dim), dtype=np.complex128)
    for col in range(dim):
        col_bits = [(col >> (n - 1 - q)) & 1 for q in range(n)]
        sub_col = 0
        for q in qubits:
            sub_col = (sub_col << 1) | col_bits[q]
        for sub_row in range(1 << k):
            amp = matrix[sub_row, sub_col]
            if amp == 0:
                continue
            row_bits = col_bits[:]
            for i, q in enumerate(qubits):
                row_bits[q] = (sub_row >> (k - 1 - i)) & 1
            row = 0
            for b in row_bits:
                row = (row << 1) | b
            full[row, col] += amp
    # macOS's Accelerate BLAS backend can raise spurious divide/overflow
    # RuntimeWarnings on complex matmul even when every operand and result
    # is finite (see qsim/gates.py:is_unitary for the same workaround).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return full @ state
