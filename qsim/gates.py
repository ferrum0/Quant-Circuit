"""
Standard gate matrices. Every gate is expressed as a small dense numpy
matrix (2x2 for single-qubit gates, 4x4/8x8 only where a dense matrix is
the natural description e.g. SWAP/Toffoli reference forms used by tests).
The simulator itself never builds a 2**n x 2**n matrix -- see qsim/state.py.
"""
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=np.complex128)

X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)          # phase(pi/2)
S_DAG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)  # phase(pi/4)
T_DAG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)


def phase(theta: float) -> np.ndarray:
    """Diagonal phase gate P(theta) = diag(1, e^{i theta})."""
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128)


def u_gate(theta: float, phi: float, lam: float) -> np.ndarray:
    """General single-qubit unitary, IBM/Qiskit U(theta, phi, lambda) convention:

        U = [[ cos(t/2)            , -e^{i lam} sin(t/2)      ],
             [ e^{i phi} sin(t/2)  ,  e^{i(phi+lam)} cos(t/2) ]]

    Any single-qubit unitary (up to global phase) can be written this way,
    so this one function covers arbitrary one-qubit gates.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


def is_unitary(matrix: np.ndarray, atol: float = 1e-8) -> bool:
    matrix = np.asarray(matrix, dtype=np.complex128)
    n = matrix.shape[0]
    # macOS's Accelerate BLAS backend can raise spurious divide/overflow
    # RuntimeWarnings on complex matmul even when every operand and result
    # is finite; suppress just around this check.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        product = matrix.conj().T @ matrix
    return np.allclose(product, np.eye(n), atol=atol)


# ---------------------------------------------------------------------
# Dense reference matrices (2**k x 2**k). Used only by tests/verify.py to
# check the optimized tensordot-based apply_gate/apply_controlled_gate
# against the textbook full-matrix definition -- never used internally by
# the simulator on more than a couple of qubits, since building these is
# the O(4**n)-style approach we are specifically avoiding.
# ---------------------------------------------------------------------
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
)
CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)
SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
)
TOFFOLI = np.eye(8, dtype=np.complex128)
TOFFOLI[[6, 7]] = TOFFOLI[[7, 6]]  # flip target bit when both controls are 1
