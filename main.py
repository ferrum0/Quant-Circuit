"""
Demo entry point for the from-scratch quantum computer simulator.

Run:  python main.py
"""
from qsim import QuantumCircuit
from qsim.visualization import draw_circuit, draw_circuit_mpl, plot_amplitudes, plot_probabilities


def main():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    qc.rz(0.7, 2)

    print("Circuit:")
    print(draw_circuit(qc))

    state = qc.statevector()
    print("\nFinal statevector:")
    print(state)

    result = qc.run(shots=5000, seed=0)
    print("\nMeasurement counts (5000 shots):")
    for bits, count in sorted(result.counts.items()):
        print(f"  |{bits}>: {count}")

    try:
        draw_circuit_mpl(qc, path="circuit.png")
        plot_probabilities(state, path="probabilities.png")
        plot_amplitudes(state, path="amplitudes.png")
        print("\nSaved circuit.png, probabilities.png, amplitudes.png")
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot generation)")


if __name__ == "__main__":
    main()
