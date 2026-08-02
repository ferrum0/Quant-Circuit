"""
Property-based tests (hypothesis). Instead of a handful of fixed examples,
these generate many random circuits/gates and check invariants that must
hold for ANY unitary gate sequence: normalization is preserved, and the
optimized apply_gate/apply_controlled_gate agree with the O(4**n) dense
reference for arbitrary Haar-random unitaries and arbitrary qubit subsets.
"""
import numpy as np
from _reference import dense_apply
from hypothesis import given, settings
from hypothesis import strategies as st

from qsim import gates as g
from qsim.circuit import QuantumCircuit
from qsim.state import QuantumState


def haar_random_unitary(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Sample a Haar-random unitary via QR decomposition of a complex
    Ginibre matrix, with the standard phase correction on R's diagonal."""
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def random_state_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return (psi / np.linalg.norm(psi)).astype(np.complex128)


@given(
    n=st.integers(min_value=1, max_value=5),
    k=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    data=st.data(),
)
@settings(max_examples=60, deadline=None)
def test_apply_gate_matches_dense_reference_random_unitaries(n, k, seed, data):
    k = min(k, n)
    qubits = data.draw(
        st.lists(st.integers(min_value=0, max_value=n - 1), min_size=k, max_size=k, unique=True)
    )
    rng = np.random.default_rng(seed)
    U = haar_random_unitary(rng, 1 << k)
    psi0 = random_state_vector(rng, 1 << n)

    expected = dense_apply(psi0, U, qubits, n)

    state = QuantumState(n)
    state.amplitudes = psi0.copy()
    state.apply_gate(U, qubits)

    assert np.allclose(state.amplitudes, expected, atol=1e-7)
    assert np.isclose(np.linalg.norm(state.amplitudes), 1.0, atol=1e-7)


@given(
    n=st.integers(min_value=2, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    data=st.data(),
)
@settings(max_examples=40, deadline=None)
def test_controlled_gate_matches_dense_reference(n, seed, data):
    n_controls = data.draw(st.integers(min_value=1, max_value=n - 1))
    all_qubits = data.draw(
        st.permutations(list(range(n))).map(lambda p: p)
    )
    controls = all_qubits[:n_controls]
    target = all_qubits[n_controls]

    rng = np.random.default_rng(seed)
    U = haar_random_unitary(rng, 2)
    psi0 = random_state_vector(rng, 1 << n)

    expected_matrix_qubits = list(controls) + [target]
    # Build the dense k=len(controls)+1 controlled-U matrix explicitly:
    # identity everywhere except the all-controls-1 block, which gets U.
    dim_sub = 1 << (n_controls + 1)
    full_sub = np.eye(dim_sub, dtype=np.complex128)
    full_sub[-2:, -2:] = U
    expected = dense_apply(psi0, full_sub, expected_matrix_qubits, n)

    state = QuantumState(n)
    state.amplitudes = psi0.copy()
    state.apply_controlled_gate(U, controls=list(controls), targets=[target])

    assert np.allclose(state.amplitudes, expected, atol=1e-7)


@given(
    n=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_gates=st.integers(min_value=0, max_value=25),
    data=st.data(),
)
@settings(max_examples=60, deadline=None)
def test_random_circuit_preserves_normalization(n, seed, n_gates, data):
    qc = QuantumCircuit(n)
    for _ in range(n_gates):
        kind = data.draw(st.sampled_from(["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cx", "ccx", "swap"]))
        if kind in ("cx", "swap"):
            a, b = data.draw(
                st.lists(st.integers(0, n - 1), min_size=2, max_size=2, unique=True)
                if n >= 2 else st.just([0, 0])
            )
            if n < 2:
                continue
            getattr(qc, kind)(a, b)
        elif kind == "ccx":
            if n < 3:
                continue
            qs = data.draw(st.lists(st.integers(0, n - 1), min_size=3, max_size=3, unique=True))
            qc.ccx(*qs)
        elif kind in ("rx", "ry", "rz"):
            q = data.draw(st.integers(0, n - 1))
            theta = data.draw(st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False))
            getattr(qc, kind)(theta, q)
        else:
            q = data.draw(st.integers(0, n - 1))
            getattr(qc, kind)(q)

    state = qc.statevector()
    assert np.isclose(np.sum(state.probabilities()), 1.0, atol=1e-7)


@given(theta=st.floats(min_value=-10, max_value=10, allow_nan=False))
@settings(max_examples=30)
def test_parametric_single_qubit_gates_are_unitary(theta):
    assert g.is_unitary(g.rx(theta))
    assert g.is_unitary(g.ry(theta))
    assert g.is_unitary(g.rz(theta))
    assert g.is_unitary(g.phase(theta))
