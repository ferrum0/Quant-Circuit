"""
Text-based and matplotlib-based visualization of circuits and state vectors.
"""
from __future__ import annotations

import numpy as np

from .circuit import QuantumCircuit
from .state import QuantumState


def draw_circuit(circuit: QuantumCircuit) -> str:
    """Render the circuit as an ASCII diagram, one column per instruction."""
    n = circuit.n
    lines = [f"q{i}: " for i in range(n)]
    pad = max(len(line) for line in lines)
    lines = [line.ljust(pad) for line in lines]

    for instr in circuit.instructions:
        col = ["──"] * n
        if instr.kind == "barrier":
            for i in range(n):
                col[i] = "░░"
        elif instr.kind == "measure":
            q = instr.qubits[0]
            col[q] = "M "
        elif instr.label in ("CX", "CZ", "CCX"):
            controls = instr.qubits[:-1]
            target = instr.qubits[-1]
            lo, hi = min(instr.qubits), max(instr.qubits)
            symbol = "X" if instr.label == "CX" else ("Z" if instr.label == "CZ" else "X")
            for i in range(lo, hi + 1):
                if i in controls:
                    col[i] = "●─"
                elif i == target:
                    col[i] = f"{symbol} " if instr.label == "CCX" else f"{symbol}─"
                else:
                    col[i] = "│─"
        elif instr.label == "SWAP":
            a, b = instr.qubits
            lo, hi = min(a, b), max(a, b)
            for i in range(lo, hi + 1):
                if i in (a, b):
                    col[i] = "x─"
                else:
                    col[i] = "│─"
        else:
            q = instr.qubits[0] if len(instr.qubits) == 1 else None
            label = instr.label[:2].ljust(2)
            if q is not None:
                col[q] = label
            else:
                for qq in instr.qubits:
                    col[qq] = label

        width = max(len(c) for c in col)
        for i in range(n):
            lines[i] += col[i].ljust(width) + "─"

    return "\n".join(lines)


def draw_circuit_mpl(circuit: QuantumCircuit, path: str | None = None, show: bool = False):
    """Render the circuit with matplotlib (rectangles for gates, dots/lines
    for controls). Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt

    n = circuit.n
    steps = circuit.instructions
    width = max(len(steps) + 1, 2)
    fig, ax = plt.subplots(figsize=(1.2 * width, 0.8 * n + 1))

    for q in range(n):
        ax.plot([0, width], [n - 1 - q, n - 1 - q], color="#888", zorder=1)
        ax.text(-0.3, n - 1 - q, f"q{q}", ha="right", va="center", fontsize=10)

    for col_idx, instr in enumerate(steps, start=1):
        if instr.kind == "barrier":
            ax.axvline(col_idx, color="#aaa", linestyle="--", zorder=0)
            continue
        if instr.kind == "measure":
            y = n - 1 - instr.qubits[0]
            box = plt.Rectangle((col_idx - 0.3, y - 0.3), 0.6, 0.6, facecolor="#f5e6a8", edgecolor="black", zorder=2)
            ax.add_patch(box)
            ax.text(col_idx, y, "M", ha="center", va="center", fontsize=9, zorder=3)
            continue
        if instr.label in ("CX", "CZ", "CCX"):
            controls = instr.qubits[:-1]
            target = instr.qubits[-1]
            ax.plot([col_idx, col_idx], [n - 1 - min(instr.qubits), n - 1 - max(instr.qubits)],
                    color="black", zorder=1)
            for c in controls:
                ax.plot(col_idx, n - 1 - c, "o", color="black", markersize=8, zorder=2)
            ty = n - 1 - target
            if instr.label == "CZ":
                ax.plot(col_idx, ty, "o", color="black", markersize=8, zorder=2)
            else:
                circle = plt.Circle((col_idx, ty), 0.25, facecolor="white", edgecolor="black", zorder=2)
                ax.add_patch(circle)
                ax.plot([col_idx - 0.25, col_idx + 0.25], [ty, ty], color="black", zorder=3)
                ax.plot([col_idx, col_idx], [ty - 0.25, ty + 0.25], color="black", zorder=3)
            continue
        if instr.label == "SWAP":
            a, b = instr.qubits
            ax.plot([col_idx, col_idx], [n - 1 - a, n - 1 - b], color="black", zorder=1)
            for q in (a, b):
                y = n - 1 - q
                ax.plot([col_idx - 0.15, col_idx + 0.15], [y - 0.15, y + 0.15], color="black", zorder=2)
                ax.plot([col_idx - 0.15, col_idx + 0.15], [y + 0.15, y - 0.15], color="black", zorder=2)
            continue
        # generic single/multi-qubit boxed gate
        for q in instr.qubits:
            y = n - 1 - q
            box = plt.Rectangle((col_idx - 0.35, y - 0.3), 0.7, 0.6, facecolor="#bcd8f2",
                                 edgecolor="black", zorder=2)
            ax.add_patch(box)
            ax.text(col_idx, y, instr.label, ha="center", va="center", fontsize=8, zorder=3)

    ax.set_xlim(-1, width + 0.5)
    ax.set_ylim(-1, n)
    ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_probabilities(state: QuantumState, path: str | None = None, show: bool = False, top: int | None = None):
    """Bar chart of measurement probabilities |amplitude_i|^2 per basis state."""
    import matplotlib.pyplot as plt

    probs = state.probabilities()
    labels = [state.bitstring(i) for i in range(state.dim)]
    order = np.argsort(probs)[::-1]
    if top is not None:
        order = order[:top]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(order)), 4))
    ax.bar([labels[i] for i in order], probs[order], color="#4c72b0")
    ax.set_ylabel("probability")
    ax.set_xlabel("basis state |q0 q1 ... qn-1>")
    ax.set_title("Measurement probabilities")
    plt.xticks(rotation=90 if len(order) > 8 else 0)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_amplitudes(state: QuantumState, path: str | None = None, show: bool = False, top: int | None = None):
    """Bar chart of |amplitude| with bar color encoding the phase (hue),
    plus a small colorbar legend for the phase angle."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable, hsv
    from matplotlib.colors import Normalize

    amps = state.amplitudes
    mags = np.abs(amps)
    phases = np.angle(amps)
    labels = [state.bitstring(i) for i in range(state.dim)]

    order = np.argsort(mags)[::-1]
    order = [i for i in order if mags[i] > 1e-10]
    if top is not None:
        order = order[:top]

    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    colors = [hsv(norm(phases[i]) if False else (phases[i] + np.pi) / (2 * np.pi)) for i in order]

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(order)), 4))
    ax.bar([labels[i] for i in order], [mags[i] for i in order], color=colors)
    ax.set_ylabel("|amplitude|")
    ax.set_xlabel("basis state")
    ax.set_title("State amplitudes (color = phase)")
    plt.xticks(rotation=90 if len(order) > 8 else 0)

    sm = ScalarMappable(cmap=hsv, norm=Normalize(vmin=-np.pi, vmax=np.pi))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("phase (radians)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig
