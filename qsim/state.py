"""
Core n-qubit state vector and the gate-application engine.

Convention
----------
Qubits are indexed 0..n-1 with qubit 0 as the MOST significant bit, i.e.
basis state |q0 q1 ... q_{n-1}> maps to the flat index

    i = sum_k q_k * 2**(n-1-k)

This is the standard "textbook" big-endian ordering and is the OPPOSITE of
Qiskit's little-endian convention (Qiskit's qubit 0 is the least significant
bit). qsim.verify accounts for this when cross-checking against Qiskit.

Representation
--------------
The state is stored as a flat length-2**n complex128 numpy array `amplitudes`.
For gate application we reshape this flat vector (without copying -- reshape
of a C-contiguous array is a view) into an n-dimensional tensor of shape
(2,)*n, where tensor axis k IS qubit k. A k-qubit gate is then applied by
contracting only the k relevant axes with the 2**k x 2**k gate matrix,
touching all 2**n amplitudes but multiplying by only 2**k entries each --
O(2**n * 2**k) instead of the naive O(4**n) full-matrix-vector product
(2**n x 2**n matrix, 2**n-length vector). See README.md for the full
derivation.
"""
from __future__ import annotations

import numpy as np


class QuantumState:
    def __init__(self, n_qubits: int, rng: np.random.Generator | None = None):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n = n_qubits
        self.dim = 1 << n_qubits
        self.amplitudes = np.zeros(self.dim, dtype=np.complex128)
        self.amplitudes[0] = 1.0  # |00...0>
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # basic utilities
    # ------------------------------------------------------------------
    def copy(self) -> QuantumState:
        new = QuantumState.__new__(QuantumState)
        new.n = self.n
        new.dim = self.dim
        new.amplitudes = self.amplitudes.copy()
        new.rng = self.rng
        return new

    def tensor(self) -> np.ndarray:
        """View of the amplitude vector as an n-dim tensor of shape (2,)*n."""
        return self.amplitudes.reshape((2,) * self.n)

    def normalize(self) -> None:
        norm = np.linalg.norm(self.amplitudes)
        if norm == 0:
            raise ValueError("zero state vector cannot be normalized")
        self.amplitudes /= norm

    def probabilities(self) -> np.ndarray:
        return np.abs(self.amplitudes) ** 2

    def bitstring(self, index: int) -> str:
        return format(index, f"0{self.n}b")

    # ------------------------------------------------------------------
    # gate application -- the O(2**n * 2**k) core routine
    # ------------------------------------------------------------------
    def apply_gate(self, matrix: np.ndarray, qubits: list[int]) -> None:
        """Apply a 2**k x 2**k unitary `matrix` to the given list of qubits.

        `qubits[i]` is the qubit that matrix row/col block `i` refers to,
        e.g. apply_gate(CNOT, [control, target]).
        """
        matrix = np.asarray(matrix, dtype=np.complex128)
        k = len(qubits)
        if matrix.shape != (1 << k, 1 << k):
            raise ValueError(f"matrix shape {matrix.shape} does not match {k} qubit(s)")
        if len(set(qubits)) != k:
            raise ValueError("duplicate qubit index in gate application")
        for q in qubits:
            if not (0 <= q < self.n):
                raise ValueError(f"qubit index {q} out of range for {self.n}-qubit state")

        T = self.tensor()
        # gate tensor: first k axes = output indices, last k axes = input indices
        U = matrix.reshape((2,) * k + (2,) * k)

        # contract U's "input" axes with T's target axes
        contracted = np.tensordot(U, T, axes=(list(range(k, 2 * k)), qubits))
        # contracted axis order: [qubits[0..k-1] (outputs)] + [T's remaining axes, ascending]
        remaining_axes = [ax for ax in range(self.n) if ax not in qubits]
        new_order = list(qubits) + remaining_axes
        inverse_perm = np.argsort(new_order)

        result = np.transpose(contracted, inverse_perm)
        self.amplitudes = np.ascontiguousarray(result).reshape(self.dim)

    def apply_controlled_gate(self, matrix: np.ndarray, controls: list[int], targets: list[int]) -> None:
        """Apply `matrix` to `targets` only on the subspace where every
        qubit in `controls` is |1>. Generalizes CNOT (1 control), Toffoli
        (2 controls), and controlled-U for any unitary/target count.

        This touches only the O(2**(n - |controls|) * 2**k) amplitudes in
        the control=1 branch; the control=0 branch is never even read.
        """
        matrix = np.asarray(matrix, dtype=np.complex128)
        k = len(targets)
        if matrix.shape != (1 << k, 1 << k):
            raise ValueError(f"matrix shape {matrix.shape} does not match {k} target qubit(s)")
        all_idx = set(controls) | set(targets)
        if len(all_idx) != len(controls) + len(targets):
            raise ValueError("controls and targets must be disjoint and each list free of duplicates")

        T = self.tensor()
        index = [slice(None)] * self.n
        for c in controls:
            index[c] = 1  # select the control=1 branch; integer index drops that axis
        sub = T[tuple(index)]  # basic indexing -> a VIEW into `self.amplitudes`

        remaining_qubits = [q for q in range(self.n) if q not in controls]
        sub_target_axes = [remaining_qubits.index(t) for t in targets]

        U = matrix.reshape((2,) * k + (2,) * k)
        contracted = np.tensordot(U, sub, axes=(list(range(k, 2 * k)), sub_target_axes))
        sub_remaining = [ax for ax in range(sub.ndim) if ax not in sub_target_axes]
        new_order = sub_target_axes + sub_remaining
        inverse_perm = np.argsort(new_order)

        sub[...] = np.transpose(contracted, inverse_perm)

    def swap(self, q1: int, q2: int) -> None:
        """Swap two qubits by permuting tensor axes -- O(2**n), no gate matrix needed."""
        if q1 == q2:
            return
        T = self.tensor()
        swapped = np.swapaxes(T, q1, q2)
        self.amplitudes = np.ascontiguousarray(swapped).reshape(self.dim)

    # ------------------------------------------------------------------
    # measurement
    # ------------------------------------------------------------------
    def prob_qubit(self, q: int) -> tuple[float, float]:
        """Marginal probability (p0, p1) of qubit q, summed over all others."""
        T = self.tensor()
        amp0 = np.take(T, indices=0, axis=q)
        p0 = float(np.sum(np.abs(amp0) ** 2))
        p1 = 1.0 - p0
        return p0, p1

    def measure(self, q: int) -> int:
        """Projective measurement of a single qubit in the Z basis.
        Collapses the state and renormalizes. Returns the classical bit."""
        p0, p1 = self.prob_qubit(q)
        p0, p1 = max(p0, 0.0), max(p1, 0.0)
        total = p0 + p1
        outcome = int(self.rng.choice([0, 1], p=[p0 / total, p1 / total]))

        T = self.tensor()
        index = [slice(None)] * self.n
        index[q] = 1 - outcome
        T[tuple(index)] = 0.0
        self.normalize()
        return outcome

    def measure_all(self) -> str:
        """Projective measurement of every qubit at once. Collapses the
        state to a single computational basis state. Returns a bitstring."""
        probs = self.probabilities()
        probs = probs / probs.sum()
        outcome = int(self.rng.choice(self.dim, p=probs))
        self.amplitudes = np.zeros(self.dim, dtype=np.complex128)
        self.amplitudes[outcome] = 1.0
        return self.bitstring(outcome)

    def sample(self, shots: int, rng: np.random.Generator | None = None) -> dict[str, int]:
        """Non-destructive statistical sampling: draws `shots` i.i.d. classical
        outcomes from the |amplitude|^2 distribution WITHOUT collapsing this
        state object and WITHOUT re-running the circuit that produced it.

        This is only valid when the state being sampled is the true final
        state of a circuit with no mid-circuit measurement/feedback -- i.e.
        the final measurement statistics depend only on the final
        statevector. qsim.circuit.QuantumCircuit.run() decides automatically
        whether this shortcut applies or whether per-shot re-simulation is
        required (see its docstring). Cost: O(2**n + shots) instead of
        O(shots * circuit_cost).
        """
        rng = rng if rng is not None else self.rng
        probs = self.probabilities()
        probs = probs / probs.sum()
        counts = rng.multinomial(shots, probs)
        return {
            self.bitstring(i): int(c)
            for i, c in enumerate(counts)
            if c > 0
        }

    def __repr__(self) -> str:
        terms = []
        for i, amp in enumerate(self.amplitudes):
            if abs(amp) > 1e-10:
                terms.append(f"({amp.real:+.4f}{amp.imag:+.4f}j)|{self.bitstring(i)}>")
        return " + ".join(terms) if terms else "0"
