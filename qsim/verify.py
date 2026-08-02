"""
Optional cross-verification against Qiskit. Qiskit is used ONLY here, as an
external oracle to sanity-check our from-scratch simulator -- never as part
of the simulator itself.

Bit-order note: qsim uses big-endian qubit ordering (qubit 0 = most
significant bit of the flat index), while Qiskit's Statevector uses
little-endian (qubit 0 = least significant bit). We permute Qiskit's
statevector into qsim's convention before comparing.
"""
from __future__ import annotations

import numpy as np

from .circuit import QuantumCircuit


def _bit_reverse_permutation(n: int) -> np.ndarray:
    dim = 1 << n
    perm = np.empty(dim, dtype=np.int64)
    for i in range(dim):
        bits = format(i, f"0{n}b")
        perm[i] = int(bits[::-1], 2)
    return perm


_SIMPLE_GATE_MAP = {
    "H": lambda qc, qs: qc.h(qs[0]),
    "X": lambda qc, qs: qc.x(qs[0]),
    "Y": lambda qc, qs: qc.y(qs[0]),
    "Z": lambda qc, qs: qc.z(qs[0]),
    "S": lambda qc, qs: qc.s(qs[0]),
    "S†": lambda qc, qs: qc.sdg(qs[0]),
    "T": lambda qc, qs: qc.t(qs[0]),
    "T†": lambda qc, qs: qc.tdg(qs[0]),
    "CX": lambda qc, qs: qc.cx(qs[0], qs[1]),
    "CZ": lambda qc, qs: qc.cz(qs[0], qs[1]),
    "CCX": lambda qc, qs: qc.ccx(qs[0], qs[1], qs[2]),
    "SWAP": lambda qc, qs: qc.swap(qs[0], qs[1]),
}


def to_qiskit_circuit(circuit: QuantumCircuit):
    from qiskit import QuantumCircuit as QiskitCircuit

    qc = QiskitCircuit(circuit.n)
    for instr in circuit.instructions:
        if instr.kind in ("barrier", "measure"):
            continue
        label, qs = instr.label, instr.qubits
        if label in _SIMPLE_GATE_MAP:
            _SIMPLE_GATE_MAP[label](qc, qs)
        elif label.startswith("P("):
            qc.p(_angle_from_matrix_phase(instr.matrix), qs[0])
        elif label.startswith("RX"):
            qc.rx(_angle_from_rotation(instr.matrix, "x"), qs[0])
        elif label.startswith("RY"):
            qc.ry(_angle_from_rotation(instr.matrix, "y"), qs[0])
        elif label.startswith("RZ"):
            qc.rz(_angle_from_rotation(instr.matrix, "z"), qs[0])
        else:
            qc.unitary(instr.matrix, qs, label=label)
    return qc


def _angle_from_matrix_phase(matrix: np.ndarray) -> float:
    return float(np.angle(matrix[1, 1]))


def _angle_from_rotation(matrix: np.ndarray, axis: str) -> float:
    if axis == "z":
        return float(-2 * np.angle(matrix[0, 0]))
    theta = 2 * np.arccos(np.clip(matrix[0, 0].real, -1, 1))
    return float(theta)


def compare_with_qiskit(circuit: QuantumCircuit, atol: float = 1e-8) -> dict:
    """Build an equivalent Qiskit circuit (skipping measurements), compute
    both statevectors, and compare them after correcting for the
    big-endian/little-endian convention difference.

    Returns a dict with 'match' (bool) and 'max_abs_diff' (float).
    Raises ImportError if Qiskit is not installed.
    """
    from qiskit.quantum_info import Statevector

    qsim_state = circuit.statevector()
    qc = to_qiskit_circuit(circuit)
    qiskit_amps = np.asarray(Statevector.from_instruction(qc).data)

    perm = _bit_reverse_permutation(circuit.n)
    qiskit_amps_reordered = qiskit_amps[perm]

    diff = np.max(np.abs(qsim_state.amplitudes - qiskit_amps_reordered))
    return {"match": bool(diff < atol), "max_abs_diff": float(diff)}
