import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Sampler
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit.quantum_info.operators import Operator
import matplotlib.pyplot as plt

from utils import cluster_solutions, elbow_plot, structural_check

# Fixed-point Grover parameter
THETA = np.pi/3

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
    # Apply oracle operations for each clause
    for i, clause in enumerate(cnf):
        get_repr(qc, False, clause, n_variables + i)

    # Mark solutions with phase
    control_qubits = list(range(n_variables, n-1))
    # Instead of using π phase, use θ phase for fixed-point Grover
    qc.mcp(THETA, control_qubits, n-1)
    qc.barrier()

    # Uncompute ancilla qubits
    for i in range(len(cnf)-1, -1, -1):
        get_repr(qc, True, cnf[i], n_variables + i)

def create_fixed_point_diffuser(n_variables):
    """Create a fixed-point diffuser operator with phase θ"""
    # Create circuit for the diffuser
    qc = QuantumCircuit(n_variables)
    
    # Apply H gates
    qc.h(range(n_variables))
    
    # Apply X gates
    qc.x(range(n_variables))
    
    # Apply controlled-phase with θ instead of π
    # For multi-qubit, we need to apply a multi-controlled phase
    control_qubits = list(range(n_variables-1))
    qc.mcp(THETA, control_qubits, n_variables-1)
    
    # Apply X gates
    qc.x(range(n_variables))
    
    # Apply H gates
    qc.h(range(n_variables))
    
    # Convert to an operator
    diffuser_op = Operator(qc)
    return diffuser_op

def fixed_point_grover_iteration(qc, n_variables, cnf, n, diffuser_op=None):
    """Apply a single fixed-point Grover iteration"""
    # Apply the oracle
    oracle(qc, n_variables, cnf, n)
    
    # Apply the fixed-point diffuser to just the variable qubits
    if diffuser_op is not None:
        qc.append(diffuser_op, range(n_variables))
    else:
        # Alternative: manually construct diffuser circuit
        qc.h(range(n_variables))
        qc.x(range(n_variables))
        control_qubits = list(range(n_variables-1))
        qc.mcp(THETA, control_qubits, n_variables-1)
        qc.x(range(n_variables))
        qc.h(range(n_variables))
    
    qc.barrier()

def recursive_fixed_point_grover(n_variables, cnf, n, iterations):
    """Implement the recursive structure of fixed-point Grover algorithm"""
    if iterations == 0:
        # Base case: Initialize circuit with Hadamards
        qc = QuantumCircuit(n)
        qc.h(range(n_variables))
        return qc
    
    if iterations == 1:
        # First iteration case
        qc = QuantumCircuit(n)
        qc.h(range(n_variables))
        
        # Apply one Grover iteration
        fixed_point_grover_iteration(qc, n_variables, cnf, n)
        return qc
    
    # Recursive case: G_k = G_{k-1} · U_0 · G_{k-1}^† · U_s · G_{k-1}
    # Where G_{k-1} is the circuit for k-1 iterations
    # U_0 is the oracle, and U_s is the diffuser
    
    # Get G_{k-1}
    prev_circuit = recursive_fixed_point_grover(n_variables, cnf, n, iterations-1)
    prev_op = Operator(prev_circuit)
    prev_op_dag = prev_op.adjoint()
    
    # Create new circuit
    qc = QuantumCircuit(n)
    
    # Apply G_{k-1}
    qc.append(prev_op, range(n))
    
    # Apply oracle (U_0)
    oracle(qc, n_variables, cnf, n)
    
    # Apply G_{k-1}^†
    qc.append(prev_op_dag, range(n))
    
    # Apply diffuser (U_s) to just the variable qubits
    diffuser_op = create_fixed_point_diffuser(n_variables)
    qc.append(diffuser_op, range(n_variables))
    
    # Apply G_{k-1}
    qc.append(prev_op, range(n))
    
    return qc

def solvePiThirdQuantumSAT(cnf, iterations=3, debug=False):
    """
    Solve a SAT problem using the fixed-point π/3 Grover algorithm.
    Parameters:
        cnf: A list of clauses in CNF form
        iterations: Number of fixed-point Grover iterations (default: 3)
        debug: Whether to print debug information and save plots
    Returns:
        is_sat: True if SAT, False if UNSAT
        solutions: List of satisfying assignments
    """
    # Perform structural checks on CNF
    structural_check(cnf)
    
    # Extract number of variables and clauses
    variables = set()
    for clause in cnf:
        for var in clause:
            variables.add(abs(var))
            
    n_variables = len(variables)
    n_clauses = len(cnf)
    
    # Total number of qubits (variables + clauses + 1 for output)
    n = n_variables + n_clauses
    
    print(f"LOG: Using {n} qubits with {iterations} fixed-point Grover iterations")
    
    # Build the fixed-point Grover circuit recursively
    qc = recursive_fixed_point_grover(n_variables, cnf, n, iterations)
    
    # Measure variable qubits
    qc.measure_all()
    
    # Debug: Save circuit diagram
    if debug:
        circuit_drawer(qc, output='mpl')
        plt.savefig('debug/circuit.png')
        plt.close()
    
    # Optimize and run the circuit
    qc = RemoveBarriers()(qc)
    optimized_qc = transpile(qc, optimization_level=3)
    result = Sampler().run([optimized_qc], shots=2048).result()
    counts = result.quasi_dists[0]
    
    counts = counts.binary_probabilities(num_bits=n)
    if debug: print(f"DEBUG: counts={counts}")
    
    # if debug: print(f"DEBUG: clustering solutions, {len(counts)}")
    # temp_counts, sil = cluster_solutions(counts)
    
    # print(f"LOG: Silhouette score: {sil}")
    
    # if debug: print(f"DEBUG: clustered solutions, {len(counts)}")
    
    # if debug: elbow_plot(counts, temp_counts)
    
    # # doing this just to debug the clustering, sorry for the mess
    # # counts = temp_counts
    
    # if debug: 
    #     plot_histogram(counts)
    #     plt.savefig('debug/histogram.png')
    #     plt.close()
    # solutions = []
    
    # if len(binary_counts) == 0:
    #     return False, []
    # else:
    #     is_sat = True
    #     # Sort by probability (highest first)
    #     for bitstring, prob in sorted(binary_counts.items(), key=lambda x: x[1], reverse=True):
    #         if debug:
    #             print(f"DEBUG: bitstring={bitstring}, prob={prob}")
            
    #         # Extract just the variable bits (first n_variables bits in the bitstring)
    #         # Note: qiskit typically reverses the bit order, so we need to handle that
    #         var_bits = bitstring[:n_variables]
    #         var_bits = var_bits[::-1]  # Reverse to get the correct ordering
            
    #         # Convert bit values to variable assignments
    #         solution = []
    #         for i, bit in enumerate(var_bits):
    #             var_num = i + 1  # Convert to 1-indexed
    #             if bit == '0':
    #                 solution.append(-var_num)
    #             else:
    #                 solution.append(var_num)
            
    #         solutions.append(solution)
    
    # return is_sat, solutions
        # create the dictionary of the counts
    dicty = {}
    
    for bistring, prob in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        
        bistring = bistring[n_clauses:]
        bistring = bistring[::-1]
        
        ## we will ad later everything.
        dicty[bistring] = prob
    
    if debug: 
        plot_histogram(counts)
        plt.savefig("debug/phi-third-histogram.png")
        plt.close()
    
    return dicty

# # Example usage:
# if __name__ == "__main__":
#     # Example CNF: (x1 OR x2) AND (NOT x1 OR x3)
#     cnf_example = [[1, 2], [-1, 3]]
    
#     # Solve using fixed-point π/3 Grover algorithm with 3 iterations
#     is_sat, solutions = solveQuantumSAT(cnf_example, iterations=3, debug=True)
    
#     if is_sat:
#         print(f"SAT problem is satisfiable.")
#         print(f"Solutions found: {solutions}")
#     else:
#         print(f"SAT problem is unsatisfiable.")