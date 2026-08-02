import numpy as np
import pytest
from _reference import dense_apply

from qsim import gates as g
from qsim.circuit import QuantumCircuit
from qsim.state import QuantumState


@pytest.mark.parametrize("n", [2, 3, 4])
def test_apply_gate_matches_dense_reference_single_qubit(n):
    rng = np.random.default_rng(0)
    for q in range(n):
        theta, phi, lam = rng.uniform(0, 2 * np.pi, 3)
        U = g.u_gate(theta, phi, lam)
        state = QuantumState(n)
        psi0 = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        psi0 /= np.linalg.norm(psi0)
        state.amplitudes = psi0.astype(np.complex128)

        expected = dense_apply(psi0, U, [q], n)
        state.apply_gate(U, [q])
        assert np.allclose(state.amplitudes, expected, atol=1e-8)


@pytest.mark.parametrize("n", [3, 4])
def test_apply_gate_matches_dense_reference_two_qubit(n):
    rng = np.random.default_rng(1)
    pairs = [(0, 1), (1, 2), (0, n - 1)]
    for a, b in pairs:
        if a >= n or b >= n:
            continue
        state = QuantumState(n)
        psi0 = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        psi0 /= np.linalg.norm(psi0)
        state.amplitudes = psi0.astype(np.complex128)

        expected = dense_apply(psi0, g.CNOT, [a, b], n)
        state.apply_gate(g.CNOT, [a, b])
        assert np.allclose(state.amplitudes, expected, atol=1e-8)


def test_controlled_gate_matches_dense_cnot():
    n = 3
    for control, target in [(0, 1), (1, 0), (0, 2), (2, 0)]:
        rng = np.random.default_rng(hash((control, target)) % (2**32))
        state = QuantumState(n)
        psi0 = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        psi0 /= np.linalg.norm(psi0)
        state.amplitudes = psi0.astype(np.complex128)

        expected = dense_apply(psi0, g.CNOT, [control, target], n)
        state.apply_controlled_gate(g.X, [control], [target])
        assert np.allclose(state.amplitudes, expected, atol=1e-8)


def test_toffoli_truth_table():
    for c1 in (0, 1):
        for c2 in (0, 1):
            for t in (0, 1):
                n = 3
                state = QuantumState(n)
                state.amplitudes[:] = 0
                idx_in = (c1 << 2) | (c2 << 1) | t
                state.amplitudes[idx_in] = 1.0
                state.apply_controlled_gate(g.X, [0, 1], [2])
                expected_t = t ^ (1 if (c1 and c2) else 0)
                idx_out = (c1 << 2) | (c2 << 1) | expected_t
                assert np.isclose(state.amplitudes[idx_out], 1.0)


def test_bell_state():
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)
    state = qc.statevector()
    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2)
    expected[3] = 1 / np.sqrt(2)
    assert np.allclose(state.amplitudes, expected, atol=1e-8)


def test_ghz_state():
    n = 4
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    state = qc.statevector()
    probs = state.probabilities()
    assert np.isclose(probs[0], 0.5, atol=1e-8)
    assert np.isclose(probs[-1], 0.5, atol=1e-8)
    assert np.isclose(probs.sum(), 1.0)


def test_swap_gate():
    n = 2
    qc = QuantumCircuit(n)
    qc.x(0)  # |10>
    qc.swap(0, 1)
    state = qc.statevector()
    assert np.isclose(state.amplitudes[1], 1.0)  # now |01>


def test_measure_all_statistics():
    n = 2
    qc = QuantumCircuit(n)
    qc.h(0).cx(0, 1)
    result = qc.run(shots=20000, seed=42)
    total = sum(result.counts.values())
    assert total == 20000
    p00 = result.counts.get("00", 0) / total
    p11 = result.counts.get("11", 0) / total
    assert abs(p00 - 0.5) < 0.02
    assert abs(p11 - 0.5) < 0.02
    assert "01" not in result.counts and "10" not in result.counts


def test_single_qubit_measure_collapse():
    n = 1
    state = QuantumState(n)
    state.amplitudes[:] = [1 / np.sqrt(2), 1 / np.sqrt(2)]
    outcome = state.measure(0)
    assert outcome in (0, 1)
    assert np.isclose(np.abs(state.amplitudes[outcome]), 1.0)
    assert np.isclose(np.abs(state.amplitudes[1 - outcome]), 0.0)


def test_mid_circuit_measurement_resimulation_path():
    n = 2
    qc = QuantumCircuit(n)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(1)  # gate AFTER measurement -> forces per-shot resimulation
    qc.measure(1, 1)
    assert qc._requires_per_shot_resimulation() is True
    result = qc.run(shots=2000, seed=1)
    for bits, count in result.counts.items():
        assert bits[1] == "1"  # qubit1 always flips to 1 regardless of qubit0
    total = sum(result.counts.values())
    assert total == 2000


def test_normalization_preserved_after_many_gates():
    n = 5
    rng = np.random.default_rng(3)
    qc = QuantumCircuit(n)
    for _ in range(30):
        gate_choice = rng.integers(0, 4)
        q = int(rng.integers(0, n))
        if gate_choice == 0:
            qc.h(q)
        elif gate_choice == 1:
            qc.rx(float(rng.uniform(0, 6.28)), q)
        elif gate_choice == 2:
            q2 = (q + 1) % n
            qc.cx(q, q2)
        else:
            qc.t(q)
    state = qc.statevector()
    assert np.isclose(np.sum(state.probabilities()), 1.0, atol=1e-8)


def test_is_unitary_helper():
    assert g.is_unitary(g.H)
    assert g.is_unitary(g.X)
    assert not g.is_unitary(np.array([[1, 1], [0, 1]]))
