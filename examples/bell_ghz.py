"""Bell pair and GHZ state examples."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsim import QuantumCircuit


def bell_pair():
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)
    return qc


def ghz_state(n=4):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


if __name__ == "__main__":
    qc = bell_pair()
    print("Bell circuit:")
    from qsim.visualization import draw_circuit
    print(draw_circuit(qc))
    state = qc.statevector()
    print("\nStatevector:", state)
    result = qc.run(shots=1000, seed=0)
    print("Counts (1000 shots):", result.counts)

    print("\n--- GHZ(4) ---")
    qc2 = ghz_state(4)
    print(draw_circuit(qc2))
    print("Statevector:", qc2.statevector())
    print("Counts:", qc2.run(shots=1000, seed=0).counts)
