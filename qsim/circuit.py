"""
QuantumCircuit: a recorded sequence of gate/measurement instructions plus
the logic to execute them against a QuantumState, including an important
optimization for statistical sampling over many shots (see run()).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import gates as g
from .state import QuantumState


@dataclass
class Instruction:
    kind: str                      # 'gate', 'measure', 'barrier'
    qubits: list[int]
    matrix: np.ndarray | None = None
    label: str = ""
    cbit: int | None = None        # classical bit index, for 'measure'


@dataclass
class CircuitResult:
    state: QuantumState
    counts: dict[str, int]

    def probabilities_from_counts(self) -> dict[str, float]:
        total = sum(self.counts.values())
        return {k: v / total for k, v in self.counts.items()}


class QuantumCircuit:
    def __init__(self, n_qubits: int, n_cbits: int | None = None):
        self.n = n_qubits
        self.n_cbits = n_cbits if n_cbits is not None else n_qubits
        self.instructions: list[Instruction] = []

    # -- single-qubit gates --------------------------------------------
    def _g1(self, matrix: np.ndarray, q: int, label: str) -> QuantumCircuit:
        self.instructions.append(Instruction("gate", [q], matrix, label))
        return self

    def h(self, q: int): return self._g1(g.H, q, "H")
    def x(self, q: int): return self._g1(g.X, q, "X")
    def y(self, q: int): return self._g1(g.Y, q, "Y")
    def z(self, q: int): return self._g1(g.Z, q, "Z")
    def s(self, q: int): return self._g1(g.S, q, "S")
    def sdg(self, q: int): return self._g1(g.S_DAG, q, "S†")
    def t(self, q: int): return self._g1(g.T_GATE, q, "T")
    def tdg(self, q: int): return self._g1(g.T_DAG, q, "T†")

    def phase(self, theta: float, q: int): return self._g1(g.phase(theta), q, f"P({theta:.2f})")
    def rx(self, theta: float, q: int): return self._g1(g.rx(theta), q, f"RX({theta:.2f})")
    def ry(self, theta: float, q: int): return self._g1(g.ry(theta), q, f"RY({theta:.2f})")
    def rz(self, theta: float, q: int): return self._g1(g.rz(theta), q, f"RZ({theta:.2f})")

    def u(self, theta: float, phi: float, lam: float, q: int):
        return self._g1(g.u_gate(theta, phi, lam), q, "U")

    def custom(self, matrix: np.ndarray, qubits: list[int], label: str = "U"):
        """Apply an arbitrary 2**k x 2**k unitary to `qubits`."""
        matrix = np.asarray(matrix, dtype=np.complex128)
        if not g.is_unitary(matrix):
            raise ValueError("custom gate matrix must be unitary")
        self.instructions.append(Instruction("gate", list(qubits), matrix, label))
        return self

    # -- multi-qubit gates -----------------------------------------------
    def cx(self, control: int, target: int):
        self.instructions.append(Instruction("gate", [control, target], g.X, "CX"))
        return self

    def cz(self, control: int, target: int):
        self.instructions.append(Instruction("gate", [control, target], g.Z, "CZ"))
        return self

    def ccx(self, c1: int, c2: int, target: int):
        self.instructions.append(Instruction("gate", [c1, c2, target], g.X, "CCX"))
        return self

    def swap(self, a: int, b: int):
        self.instructions.append(Instruction("gate", [a, b], None, "SWAP"))
        return self

    def barrier(self):
        self.instructions.append(Instruction("barrier", []))
        return self

    def measure(self, q: int, c: int | None = None):
        self.instructions.append(Instruction("measure", [q], cbit=c if c is not None else q))
        return self

    def measure_all(self):
        for q in range(self.n):
            self.measure(q, q)
        return self

    # -- execution ---------------------------------------------------
    def _apply_instruction(self, state: QuantumState, instr: Instruction, cbits: dict[int, int]) -> None:
        if instr.kind == "barrier":
            return
        if instr.kind == "measure":
            outcome = state.measure(instr.qubits[0])
            cbits[instr.cbit] = outcome
            return
        # gate
        if instr.label == "SWAP":
            state.swap(instr.qubits[0], instr.qubits[1])
        elif instr.label in ("CX", "CZ"):
            state.apply_controlled_gate(instr.matrix, controls=[instr.qubits[0]], targets=[instr.qubits[1]])
        elif instr.label == "CCX":
            state.apply_controlled_gate(instr.matrix, controls=instr.qubits[:2], targets=[instr.qubits[2]])
        else:
            state.apply_gate(instr.matrix, instr.qubits)

    def _run_once(self, rng: np.random.Generator) -> tuple[QuantumState, dict[int, int]]:
        state = QuantumState(self.n, rng=rng)
        cbits: dict[int, int] = {}
        for instr in self.instructions:
            self._apply_instruction(state, instr, cbits)
        return state, cbits

    def _requires_per_shot_resimulation(self) -> bool:
        """True if any measurement is followed by further gates (mid-circuit
        measurement with potential downstream dependence). In that case the
        classical outcome distribution is not a simple function of a single
        final statevector, so each shot must re-run the whole circuit.

        If every measurement instruction is a trailing "measure at the end"
        (no gate instructions occur after the first measurement), the final
        statevector fully determines the outcome distribution and we can use
        the O(2**n + shots) multinomial-sampling shortcut instead.
        """
        first_measure_idx = next(
            (i for i, instr in enumerate(self.instructions) if instr.kind == "measure"), None
        )
        if first_measure_idx is None:
            return False
        tail = self.instructions[first_measure_idx:]
        return any(instr.kind == "gate" for instr in tail)

    def run(self, shots: int = 1024, seed: int | None = None) -> CircuitResult:
        """Execute the circuit and collect measurement statistics over
        `shots` runs.

        Fast path (no mid-circuit measurement dependence): simulate the
        circuit ONCE to obtain the final statevector, then draw `shots`
        i.i.d. classical samples from |amplitude|^2 via a single multinomial
        draw -- O(2**n + shots) total, instead of O(shots * circuit_cost).

        Slow path (mid-circuit measurement followed by more gates): outcomes
        can depend on earlier collapses, so the circuit is re-executed
        independently `shots` times -- O(shots * circuit_cost).
        """
        rng = np.random.default_rng(seed)

        if not self._requires_per_shot_resimulation():
            state, cbits = self._run_once(rng)
            if any(instr.kind == "measure" for instr in self.instructions):
                # trailing measurements only select which qubits are reported
                measured_qubits = sorted({instr.qubits[0] for instr in self.instructions if instr.kind == "measure"})
                probs = state.probabilities()
                agg: Counter[str] = Counter()
                for i, p in enumerate(probs):
                    full_bits = state.bitstring(i)
                    key = "".join(full_bits[q] for q in measured_qubits)
                    agg[key] += p
                counts_arr = rng.multinomial(shots, list(agg.values()))
                counts = {k: int(c) for k, c in zip(agg.keys(), counts_arr) if c > 0}
            else:
                counts = state.sample(shots, rng)
            return CircuitResult(state, counts)

        counts = Counter()
        final_state = None
        for _ in range(shots):
            final_state, cbits = self._run_once(rng)
            bitstring = "".join(str(cbits.get(i, 0)) for i in range(self.n_cbits))
            counts[bitstring] += 1
        return CircuitResult(final_state, dict(counts))

    def statevector(self) -> QuantumState:
        """Run the circuit once (ignoring any measurement collapse effect on
        sampling, but still executing measurements in-order) and return the
        resulting state -- useful for inspecting amplitudes directly."""
        state, _ = self._run_once(np.random.default_rng())
        return state
