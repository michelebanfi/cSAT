import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Sampler
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.visualization import circuit_drawer

import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

from utils import cluster_solutions, elbow_plot, structural_check

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

def oracle(qc, n_variables, cnf, n):
    # print(f"DEBUG: n_variables={n_variables}, cnf={cnf}, n={n}")
    for i, clause in enumerate(cnf):
        # print(f"DEBUG: clause={clause}, i={i}, n_variables+i={n_variables+i}")
        get_repr(qc, False, clause, n_variables + i)

    qc.mcp(np.pi, list(range(n_variables,n-1)), n-1)
    qc.barrier()

    # Uncompute ancilla qubits
    for i in range(len(cnf)-1, -1, -1):
        get_repr(qc, True, cnf[i], n_variables + i)

def diffuser(qc, n):
    qc.h(range(n))
    qc.x(range(n))
    qc.mcp(np.pi, list(range(n-1)), n - 1)
    qc.x(range(n))
    qc.h(range(n))
    qc.barrier()

def create_circuit(qc, n_variables, cnf, n):
    oracle(qc, n_variables, cnf, n)
    diffuser(qc, n_variables)

def solveQuantumSAT(cnf, debug=False):
    
    # print qiskit version
    # print(f"LOG: qiskit version: {qiskit.__version__}")
    
    # print(f"LOG: creating circuit")
    
    structural_check(cnf)
    
    variables = set()
    for clause in cnf:
        for var in clause:
            variables.add(abs(var))
            
    n_variables = len(variables)
    n_clauses = len(cnf)
    
    n = n_variables + n_clauses
    
    if debug: print(f"DEBUG: {np.pi/4 * math.sqrt(2**n_variables)} reps")
    
    reps = math.ceil(np.pi/4 * math.sqrt(2**n_variables))
    
    qc = QuantumCircuit(n)
    
    qc.h(list(range(n_variables)))
    
    for i in range(reps):
        create_circuit(qc, n_variables, cnf, n)
        
    # Measure variable qubits
    qc.measure_all()
    
    # uncomment just for debugging the circuit (yes, i mean the plotting)
    if debug:  
        circuit_drawer(qc, output='mpl')
        plt.show()
    
    # remove barriers from the circuit
    qc = RemoveBarriers()(qc)
    optimized_qc = transpile(qc, optimization_level=3)
    result = Sampler().run([optimized_qc], shots=2048).result()
    if debug: print(f"DEBUG: result={result}")
    
    counts = result.quasi_dists[0]
    
    if debug: print(f"DEBUG: counts={counts}")
    
    counts = counts.binary_probabilities(num_bits=n)
    if debug: print(f"DEBUG: counts={counts}")
    
    if debug: elbow_plot(counts)
    
    if debug: print(f"DEBUG: clustering solutions, {len(counts)}")
    counts = cluster_solutions(counts)
    if debug: print(f"DEBUG: clustered solutions, {len(counts)}")
    
    if debug: 
        plot_histogram(counts)
        plt.show()
    solutions = []
    
    if len(counts) == 0:
        return False, []
    else:
        is_sat = True
        for bitstring, prob in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            # print(f"DEBUG: bitstring={bitstring}, prob={prob}")
            # remove the first n_clauses bits
            bitstring = bitstring[n_clauses:]
            # print(f"DEBUG: bitstring={bitstring}")
            # reverse the bitstring ordering
            bitstring = bitstring[::-1]
            # print(f"DEBUG: bitstring={bitstring}")
            solution = []
            for i in range(n_variables):
                # print(i)
                var_num = i + 1  # Convert to 1-indexed
                if bitstring[i] == '0':
                    solution.append(-var_num)
                else:
                    solution.append(var_num)
            solutions.append(solution)
    
    # print(f"DEBUG: solutions={solutions}")
        
    return is_sat, solutions