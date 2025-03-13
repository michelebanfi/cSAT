import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.visualization import circuit_drawer
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram

from utils import structural_check

def chebyshev(n, x):
    if abs(x) <= 1:
        return np.cos(n * np.arccos(x))
    else:
        return np.cosh(n * np.arccosh(x))

def get_repr(qc, is_inv, clause, i):
    # Track which qubits we need to flip back later
    flipped_qubits = []
    
    # For OR logic in CNF, we need to detect when the clause is NOT satisfied
    # This happens when all literals in the clause are false
    for var in clause:
        var_idx = abs(var) - 1  # Convert to 0-indexed
        
        # For positive literals, we want to detect when they're false
        # For negative literals, we want to detect when they're true
        if var > 0:  # Only flip POSITIVE variables
            qc.x(var_idx)
            flipped_qubits.append(var_idx)
    
    # Control qubits for the multi-controlled X gate
    control_qubits = [abs(var) - 1 for var in clause]
    
    # For uncomputing, invert the ancilla if needed
    if is_inv:
        qc.x(i)
    
    # Apply multi-controlled-X to detect when all literals evaluate to false
    qc.mcx(control_qubits, i)
    
    # For computing, invert the ancilla to make it 1 when clause is satisfied
    if not is_inv:
        qc.x(i)
    
    # Restore the original state of the qubits
    for var_idx in flipped_qubits:
        qc.x(var_idx)
    
    qc.barrier()
    
def oracle(qc, n_variables, beta, cnf, n):
    # print(f"DEBUG: n_variables={n_variables}, cnf={cnf}, n={n}")
    for i, clause in enumerate(cnf):
        # print(f"DEBUG: clause={clause}, i={i}, n_variables+i={n_variables+i}")
        get_repr(qc, False, clause, n_variables + i)

    qc.mcp(beta, list(range(n_variables,n-1)), n-1)
    qc.barrier()

    # Uncompute ancilla qubits
    for i in range(len(cnf)-1, -1, -1):
        get_repr(qc, True, cnf[i], n_variables + i)

def diffuser(qc, n, alpha):
    qc.h(range(n))
    qc.x(range(n))
    qc.mcp(alpha, list(range(n-1)), n - 1)
    qc.x(range(n))
    qc.h(range(n))
    qc.barrier()

def create_circuit(qc, n_variables, cnf, n, alpha, beta):
    oracle(qc, n_variables, beta,  cnf, n)
    diffuser(qc, n_variables, alpha)

def createCircuit(n_variables, l_iterations, cnf, n, delta = 0.1):
    qc = QuantumCircuit(n)
    
    qc.h(list(range(n_variables)))
    qc.barrier()
    
    # L = 2l + 1
    L = 2 * l_iterations + 1
    
    gamma_inv = chebyshev(1/L, 1/delta)
    
    
    print("ALPHA               BETA")
    
    for i in range(1, l_iterations + 1):
        alpha_i = 2 * np.arctan2(1, np.tan(2*np.pi * i / L)) * np.sqrt(1 - 1/(gamma_inv**2))
        beta_i = -2 * np.arctan2(1, np.tan(2*np.pi *(L-i) / L)) * np.sqrt(1 - 1/(gamma_inv**2))
        
        print(alpha_i, beta_i)
    
        create_circuit(qc, n_variables, cnf, n, alpha_i, beta_i)
    
    qc.barrier()
    
    return qc

def solveFixedQuantunSAT(cnf, debug=False):
    
    # as usual structural check for the CNF
    structural_check(cnf)
    
    variables = set()
    for clause in cnf:
        for var in clause:
            variables.add(abs(var))
            
    n_variables = len(variables)
    n_clauses = len(cnf)
    
    n = n_variables + n_clauses
    
    l_iterations = int(np.ceil(np.sqrt(2**n_variables) / 4))
    print(f"LOG: using {n_variables} variables and {l_iterations} iterations")
    qc = createCircuit(n_variables, l_iterations, cnf, n)
    
    qc.measure_all()
    
    if debug:
        circuit_drawer(qc, output="mpl")
        plt.savefig("debug/fixed-scircuit.png")
        plt.close()
    
    qc = RemoveBarriers()(qc)
    optimized_qc = transpile(qc, optimization_level=3)
    
    result = Sampler().run([optimized_qc], shots=1024).result()
    
    counts = result.quasi_dists[0]
    
    counts = counts.binary_probabilities(num_bits=n)
    
    if debug: 
        plot_histogram(counts)
        plt.savefig("debug/fixed-histogram.png")
        plt.close()