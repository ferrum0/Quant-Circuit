# qsim — a quantum computer simulator built from scratch (NumPy, no Qiskit)

A state-vector quantum computer simulator implemented entirely on top of
NumPy. Qiskit is used only in `qsim/verify.py` as an external oracle for
cross-checking results — the simulator itself never imports it.

## Running it

Dependencies are declared in `pyproject.toml` (single source of truth — no
separate requirements.txt). `numpy` is the only hard dependency; everything
else is an optional extra.

```bash
pip install -e ".[dev,viz]"   
python3 main.py                
pytest -v                      
ruff check .                  
python3 examples/bell_ghz.py
python3 examples/grover.py
```

## Development

- **CI**: every push/PR runs `.github/workflows/ci.yml` — `ruff check .` +
  `pytest` on Python 3.9 through 3.12.
- **Property-based tests** (`tests/test_properties.py`, via `hypothesis`):
  instead of only a handful of fixed examples, these generate random `n`,
  random qubit subsets, and Haar-random unitaries (sampled via QR
  decomposition of a complex Ginibre matrix) and check that `apply_gate` /
  `apply_controlled_gate` agree with the independent `O(4^n)` reference in
  `tests/_reference.py`, and that normalization survives arbitrary random
  gate sequences.
- **Packaging**: `pyproject.toml` (setuptools backend) makes `qsim`
  `pip install -e`-able; `tests/conftest.py` also puts the repo root on
  `sys.path` so the test suite runs against a checkout with no install step.

---

## 1. Class architecture

```
qsim/
├── state.py          QuantumState   — state vector + gate-application engine
├── gates.py                          — gate matrices (2x2, parametric generators)
├── circuit.py         QuantumCircuit — instruction list, execution, statistics
│                       CircuitResult — statevector + measurement counts
├── visualization.py                  — ASCII diagram, matplotlib diagram, amplitude plots
└── verify.py                         — cross-check against Qiskit (optional)
```

**`QuantumState`** is the sole holder of the physics. It stores
`amplitudes`: a flat `complex128` array of length `2^n`. Every operation
(applying a gate, measuring, sampling) is a method on this class. It knows
nothing about "circuits" — only about the current state vector.

**`QuantumCircuit`** is a purely declarative layer: a list of `Instruction`
objects (gate/measurement/barrier). It doesn't hold state itself — it
builds a fresh `QuantumState` on every run (`_run_once`). This separation
matters for repeated sampling (section 4).

**`visualization.py`** and **`verify.py`** depend on the two modules above,
never the other way around — the simulator works fine without
matplotlib/qiskit installed.

---

## 2. Math: tensor products and applying a gate to a subset of qubits

### 2.1 The Hilbert space of n qubits

The state of n qubits lives in `H = (C^2)^⊗n`, a space of dimension `2^n`.
The basis is the computational basis `|q0 q1 ... q_{n-1}>`, where each
`q_k ∈ {0,1}`. The state vector `|ψ> = Σ_i c_i |i>` is simply `2^n` complex
numbers `c_i`. In the code, `i` is the index into `amplitudes`, and bit `k`
of `i` (using big-endian numbering, qubit 0 = most significant bit) is the
value of qubit `k`.

### 2.2 Applying a gate to a subset of qubits — the tensor formula

If a gate `U` (a unitary `2^k × 2^k` matrix) acts on qubits
`(t_1, ..., t_k)` out of `n`, then formally the operator acting on the
whole space is

```
U_full = P · (U ⊗ I_{2^(n-k)}) · P^{-1}
```

where `P` is a basis permutation that moves the target qubits to the
front. Written out explicitly as an operator on `(C^2)^⊗n`:

```
U_full = insert U at positions (t_1,...,t_k),
         I2 at the remaining n-k positions, in the qubits' original order.
```

For example, `H` on qubit 1 out of 3 (`t=[1]`): `U_full = I2 ⊗ H ⊗ I2`
(an `8×8` matrix). CNOT on qubits `(0,2)` out of 3 can't be written as a
plain `⊗` because the qubits aren't adjacent — you need the `CNOT` matrix
"spread out" across non-adjacent positions via basis reindexing. That
reindexing is precisely the main technical headache of a direct `⊗`-based
implementation.

**The naive implementation** builds `U_full` explicitly (a dense or sparse
`2^n × 2^n` matrix via a sequence of `kron()` calls), then computes
`U_full @ ψ`. This is exactly what we avoid.

### 2.3 Why that's O(4^n): complexity of the naive approach

- Building `U_full` via `kron`: the result is a `2^n × 2^n` matrix, i.e.
  `4^n` entries. Materializing it alone is already `O(4^n)` in both memory
  and time.
- Multiplying a dense `2^n × 2^n` matrix by a length-`2^n` vector:
  `2^n` rows × `2^n` columns = `O(4^n)` multiply-adds.
- Even with a sparse representation (a gate is only sparse in special
  cases like CNOT/Z; for a general dense `U` on `k` qubits, sparsity
  doesn't help), the equivalent work of reindexing the basis is still
  linear in `4^n` in the general worst case of building the full operator.

For `n=20` that's `4^20 ≈ 10^12` — already infeasible, even though the
state vector itself (`2^20 ≈ 10^6` complex numbers, ~16 MB) fits in memory
comfortably.

### 2.4 The optimization: reshape into a rank-n tensor + `tensordot` on only the relevant axes

**Key observation**: `(C^2)^⊗n ≅ C^{2×2×...×2}` (n axes). Instead of a
flat vector `amplitudes[i]`, treat it as a rank-n tensor
`T[q0, q1, ..., q_{n-1}]` — this is just a `reshape` with no data copy (for
a C-contiguous array, reshape is a view, `O(1)`):

```python
T = self.amplitudes.reshape((2,) * self.n)   # T.shape = (2,2,...,2), n times
```

Applying `U` (which acts on axes `t_1,...,t_k`) is a **tensor
contraction** over only those `k` axes:

```
result[t_1,...,t_k; rest] = Σ_{j_1..j_k} U[t_1..t_k, j_1..j_k] · T[j_1..j_k; rest]
```

In code (`qsim/state.py: apply_gate`) this is exactly `np.tensordot`:

```python
U_tensor = matrix.reshape((2,)*k + (2,)*k)          # (out_1..out_k, in_1..in_k)
contracted = np.tensordot(U_tensor, T,
                           axes=(list(range(k, 2*k)), qubits))  # contract U's "in" axes with T's qubit axes
```

`tensordot` never builds a `2^n × 2^n` matrix — it directly sums over `k`
axes for each of the `2^(n-k)` combinations of the remaining indices.
Afterwards the axes need to be moved back to their original positions
(`np.transpose` with the inverse permutation) — this is exactly what the
permutation `P` in the §2.2 formula used to require, except now it's done
**on tensor indices**, not by multiplying huge matrices together.

### 2.5 Complexity of the optimized approach: O(2^n · 2^k)

- The output tensor has `2^k` (new axes) `× 2^(n-k)` (untouched axes) `=
  2^n` elements — exactly the size of the state, nothing extra is ever
  created.
- Producing **each** of the `2^n` output elements requires summing `2^k`
  terms (per the contraction formula above).
- Total: `2^n × 2^k` multiply-add operations → **O(2^n · 2^k)**.

For gates with constant `k` (1, 2, 3 — Hadamard, CNOT, Toffoli) this is
**O(2^n)** — linear in the size of the state, an exponential win over
`O(4^n)`. At `n=20, k=2`: naively `~10^12` operations, optimized
`~4×10^6` — a difference of six orders of magnitude.

| n  | naive O(4^n)        | optimized O(2^n·2^k), k=2 |
|----|----------------------|----------------------------|
| 10 | ~10^6                | ~4×10^3                    |
| 15 | ~10^9                | ~1.3×10^5                  |
| 20 | ~10^12               | ~4×10^6                    |
| 25 | ~10^15 (infeasible)  | ~1.3×10^8 (seconds)        |

### 2.6 Controlled gates (CNOT, Toffoli) — one more level of optimization

For `apply_controlled_gate` we don't apply the gate to the whole space at
all — the control qubits are **pinned** via integer indexing of the tensor
(`T[..., 1, ...]`, not a `slice`), which in NumPy yields a **view**, not a
copy:

```python
index = [slice(None)] * n
for c in controls:
    index[c] = 1        # select only the "control = 1" branch
sub = T[tuple(index)]   # a view — no copy!
```

`tensordot` is then applied only to this sub-tensor of size
`2^(n-|controls|)`, and the result is written back with
`sub[...] = ...` — modifying `self.amplitudes` **in place**. The
`control=0` branch is never read or written at all. The resulting
complexity for Toffoli (2 controls, 1 target) is
`O(2^(n-2) · 2^1) = O(2^n / 2)` — even better than the general
`O(2^n·2^k)` bound.

---

## 3. Gate set

| Gate | File/method | Matrix |
|---|---|---|
| Hadamard | `gates.H`, `circuit.h(q)` | `1/√2 [[1,1],[1,-1]]` |
| Pauli X/Y/Z | `gates.X/Y/Z` | standard |
| S, T (and daggers) | `gates.S/T/S_DAG/T_DAG` | `phase(π/2)`, `phase(π/4)` |
| Phase(θ) | `gates.phase(theta)` | `diag(1, e^{iθ})` |
| RX/RY/RZ(θ) | `gates.rx/ry/rz` | Bloch-sphere rotations |
| Arbitrary U(θ,φ,λ) | `gates.u_gate` | covers any single-qubit unitary (IBM convention) |
| CNOT | `circuit.cx` → `apply_controlled_gate(X, [c], [t])` | |
| CZ | `circuit.cz` → `apply_controlled_gate(Z, [c], [t])` | |
| Toffoli (CCX) | `circuit.ccx` → `apply_controlled_gate(X, [c1,c2], [t])` | |
| SWAP | `circuit.swap` → tensor axis permutation | no matrix needed, `O(2^n)` |
| Arbitrary k-qubit unitary | `circuit.custom(matrix, qubits)` | checked for unitarity |

`apply_controlled_gate` is generalized to **any** number of controls and
targets, so controlled-U with an arbitrary number of controls comes "for
free" — there's no need to write a separate class for each case.

---

## 4. Measurement and statistical sampling

### 4.1 Projective measurement of a single qubit (`QuantumState.measure`)

1. The marginal probability `p1 = Σ |amplitude|^2` over all basis states
   where qubit `q`'s bit equals 1 (implemented as `np.take` along axis `q`
   of the tensor — again, without ever building a `2^n×2^n` projector).
2. An outcome `outcome ∈ {0,1}` is sampled according to `(p0, p1)`.
3. **Collapse**: amplitudes in the "incompatible" branch are zeroed
   (`T[..., 1-outcome, ...] = 0`), and the vector is renormalized by
   dividing by `√p_outcome`.

### 4.2 Measuring all qubits (`measure_all`)

An index `i ∈ [0, 2^n)` is sampled directly from the distribution
`|c_i|^2`, and the state collapses to the pure basis vector `|i>`.

### 4.3 Repeated sampling (shots) — a key optimization

The naive approach to "run the circuit 1000 times and gather statistics"
is to literally re-simulate the whole circuit 1000 times:
`O(shots · n · 2^n)`.

**Optimization (`QuantumState.sample`, `QuantumCircuit.run`)**: if the
circuit has no measurement *in the middle* followed by more gates (i.e. no
classical feedback), the final outcome distribution is a deterministic
function of a **single** final state vector. Then:

- the circuit is simulated **once** → we get `|ψ_final>`;
- `shots` classical outcomes are sampled with a single call to
  `rng.multinomial(shots, |ψ_final|^2)` — `O(2^n + shots)` instead of
  `O(shots · 2^n)`.

`QuantumCircuit._requires_per_shot_resimulation()` detects whether any
gates occur after the first measurement — if so (mid-circuit measurement
with outcome-dependent downstream gates), the honest "slow path" kicks in:
each of the `shots` runs is re-simulated independently, because the result
of such a run depends on which branch the intermediate measurement
collapsed into. The test `test_mid_circuit_measurement_resimulation_path`
checks both modes.

---

## 5. Visualization

- `visualization.draw_circuit(qc)` — an ASCII diagram (fast, no
  dependencies):
  ```
  q0: H ─●──●─────
  q1: ───X──●─────
  q2: ──────X ─RZ─
  ```
- `visualization.draw_circuit_mpl(qc, path=...)` — a matplotlib rendering:
  boxes for single-qubit gates, dots + a vertical line for controls, ⊕ for
  CNOT/Toffoli targets, crosses for SWAP.
- `visualization.plot_probabilities(state)` — a bar chart of `|c_i|^2`
  over basis states.
- `visualization.plot_amplitudes(state)` — bars of height `|c_i|`, with
  the bar color encoding the phase `arg(c_i)` (HSV colormap) — so a single
  plot shows both magnitude and phase of each amplitude.

---

## 6. Verification

`qsim/verify.py:compare_with_qiskit(circuit)` builds an equivalent Qiskit
circuit (from your instruction list), gets `Statevector.from_instruction`,
and compares it against our state vector **correcting for qubit
ordering**: qsim uses big-endian (qubit 0 = most significant bit), Qiskit
uses little-endian (qubit 0 = least significant bit), so indices are
bit-reversed before comparison (`_bit_reverse_permutation`). Qiskit takes
no part in the simulator itself — this is purely an external correctness
check, and requires `pip install qiskit`.

```python
from qsim import QuantumCircuit
from qsim.verify import compare_with_qiskit

qc = QuantumCircuit(3).h(0).cx(0,1).ccx(0,1,2)
print(compare_with_qiskit(qc))  # {'match': True, 'max_abs_diff': ~1e-16}
```

---

## 7. Tests

19 tests total, all passing, run via `pytest -v` (also in CI on Python
3.9–3.12):

`tests/test_qsim.py` — example-based:

- **Correctness of the optimization**: `apply_gate`/`apply_controlled_gate`
  are checked against an independent `O(4^n)` reference implementation
  (`dense_apply` in `tests/_reference.py`), which builds the full operator
  via explicit bit bookkeeping — for random unitary gates on 1–2 qubits in
  systems of 2–4 qubits.
- Toffoli truth table across all 8 input combinations.
- Bell/GHZ states — exact amplitudes.
- SWAP, and collapse on a single-qubit measurement.
- Measurement statistics (Bell pair, 20000 shots, deviation < 2%).
- Both `run()` modes: fast sampling and per-shot resimulation under
  mid-circuit measurement.
- Normalization preserved after 30 random gates on 5 qubits.

`tests/test_properties.py` — property-based (via `hypothesis`), generating
many random instances per property instead of a handful of fixed examples:

- `apply_gate` matches the dense reference for **Haar-random** unitaries
  (sampled via QR decomposition of a complex Ginibre matrix) on random
  qubit subsets, for random `n` up to 5.
- `apply_controlled_gate` matches an explicitly-constructed dense
  controlled-U for a random number of controls/targets.
- Normalization survives random circuits of up to 25 gates drawn from the
  full gate set, for random `n` up to 5.
- RX/RY/RZ/Phase stay unitary for arbitrary real angles.

---

## 8. What's next (if scaling further is needed)

- **Density matrices / decoherence**: the current simulator is pure-state
  (state-vector) only. Mixed states and noise would need a density matrix
  `ρ` (`2^n × 2^n`) or Kraus operators — a separate `DensityMatrix` class
  built on the same tensor-contraction idea.
- **A C++/pybind11 core**: the `apply_gate` algorithm itself (reshape +
  tensordot + transpose) ports to C++ almost verbatim — instead of NumPy
  you'd use a manual loop over bitmasks (see §2.6; the idea of "pinning
  control bits via indexing" maps naturally onto bitwise operations like
  `i & mask`). The payoff is control over allocations and
  multithreading (OpenMP over the outer `2^(n-k)` loop); the class
  architecture wouldn't need to change.
- **Multi-controlled gate decomposition**: `examples/grover.py`
  deliberately uses full `2^n × 2^n` oracle/diffusion matrices (`k=n`) as
  a clear illustration of the `O(4^n)` worst case — a real Grover
  implementation decomposes these into `O(n)` elementary gates using
  ancilla qubits.
