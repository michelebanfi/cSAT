import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Sampler
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.visualization import circuit_drawer
from qiskit.quantum_info import Operator

import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

from utils import cluster_solutions, elbow_plot, structural_check

def get_repr(qc, is_inv, clause, i):
    flipped_qubits = []
    for var in clause:
        var_idx = abs(var) - 1
        if var > 0:
            qc.x(var_idx)
            flipped_qubits.append(var_idx)

    control_qubits = [abs(var) - 1 for var in clause]

    if is_inv:
        qc.x(i)

    # Apply multi-controlled phase shift of pi/3
    qc.mcx(control_qubits, i)

    if not is_inv:
        qc.x(i)

    for var_idx in flipped_qubits:
        qc.x(var_idx)

    qc.barrier()

def oracle(qc, n_variables, cnf, n):
    for i, clause in enumerate(cnf):
        get_repr(qc, False, clause, n_variables + i)

    qc.mcp(np.pi/3, list(range(n_variables, n - 1)), n - 1)
    qc.barrier()

    for i in range(len(cnf) - 1, -1, -1):
        get_repr(qc, True, cnf[i], n_variables + i)
        
def diffuser(qc, n):
    # qc.h(range(n))
    # qc.x(range(n))
    qc.mcp(np.pi / 3, list(range(n - 1)), n - 1)
    # qc.x(range(n))
    # qc.h(range(n))
    qc.barrier()

def create_transformation_circuit(n_variables, cnf, n, m):
    
    if m == 1:
        qc = QuantumCircuit(n)
        oracle(qc, n_variables, cnf, n)
        
        qc.h(range(n))
        diffuser(qc, n_variables)
        qc.h(range(n))
        return qc
    else:
        qc_prev = create_transformation_circuit(n_variables, cnf, n, m - 1)
        qc = QuantumCircuit(n)

        # Recursive call
        qc = qc.compose(qc_prev)

        # Apply U R_s U^\dagger R_t U
        oracle(qc, n_variables, cnf, n)
        diffuser(qc, n_variables)

        # Convert to Operator and compute adjoint
        op_prev = Operator(qc_prev)
        op_inv = op_prev.adjoint()

        # Convert back to QuantumCircuit
        qc_inv = op_inv.to_instruction().definition

        qc = qc.compose(qc_inv)
        diffuser(qc, n_variables)
        oracle(qc, n_variables, cnf, n)

        return qc


def create_circuit(qc, n_variables, cnf, n, m):
    transformation_circuit = create_transformation_circuit(n_variables, cnf, n, m)
    qc.compose(transformation_circuit, inplace=True)

def solvePiThirdQuantumSAT(cnf, reps, debug=False):
    structural_check(cnf)

    variables = set()
    for clause in cnf:
        for var in clause:
            variables.add(abs(var))

    n_variables = len(variables)
    n_clauses = len(cnf)

    n = n_variables + n_clauses

    # reps = math.ceil(np.pi / 4 * math.sqrt(2**n_variables))

    qc = QuantumCircuit(n)

    qc.h(list(range(n_variables)))

    for i in range(reps):
        create_circuit(qc, n_variables, cnf, n, reps)

    qc.measure_all()

    if debug:
        circuit_drawer(qc, output='mpl')
        plt.savefig('debug/circuit.png')
        plt.close()

    qc = RemoveBarriers()(qc)
    optimized_qc = transpile(qc, optimization_level=3)
    result = Sampler().run([optimized_qc], shots=1024).result()

    counts = result.quasi_dists[0]
    counts = counts.binary_probabilities(num_bits=n)
    
    return counts

    # temp_counts, sil = cluster_solutions(counts)

    # print(f"LOG: Silhouette score: {sil}")

    # if debug:
    #     elbow_plot(counts, temp_counts)

    # if debug:
    #     plot_histogram(counts)
    #     plt.savefig('debug/histogram.png')
    #     plt.close()

    # solutions = []

    # if len(counts) == 0:
    #     return False, []
    # else:
    #     is_sat = True
    #     for bitstring, prob in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    #         bitstring = bitstring[n_clauses:][::-1]
    #         solution = []
    #         for i in range(n_variables):
    #             var_num = i + 1
    #             if bitstring[i] == '0':
    #                 solution.append(-var_num)
    #             else:
    #                 solution.append(var_num)
    #         solutions.append(solution)

    # return is_sat, solutions
