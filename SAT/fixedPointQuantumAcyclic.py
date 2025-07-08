import numpy as np
import mpmath as mpm
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile, AncillaRegister, QuantumRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from utils import structural_check, elbow_plot, cluster_solutions

def chebyshev(L, x):
    return mpm.cos(L * mpm.acos(x))

def build_cnf_oracle(cnf, n_variables, n_clauses):
    """Builds the CNF checking logic as a separate, reversible circuit."""
    edge_q = QuantumRegister(n_variables, name='edge')
    clause_a = AncillaRegister(n_clauses, name='clause')
    cnf_ok_a = AncillaRegister(1, name='cnf_ok')
    qc = QuantumCircuit(edge_q, clause_a, cnf_ok_a)

    # For each clause, check if it's UNSATISFIED
    for i, clause in enumerate(cnf):
        for literal in clause:
            if literal > 0:
                qc.x(edge_q[abs(literal) - 1])
        
        control_indices = [edge_q[abs(literal) - 1] for literal in clause]
        qc.mcx(control_indices, clause_a[i])

        for literal in clause:
            if literal > 0:
                qc.x(edge_q[abs(literal) - 1])
    
    # CNF is OK if all clause ancillas are 0
    qc.x(clause_a)
    qc.mcx(clause_a[:], cnf_ok_a[0])
    qc.x(clause_a)
    
    return qc

def build_acyclicity_oracle(num_nodes, node_names, causal_dict, reverse_cnf_map, n_variables):
    """
    Builds the acyclicity oracle based on Kahn's algorithm as a separate,
    reversible circuit.
    """
    # Define the necessary quantum registers for this specific oracle
    edge_q = QuantumRegister(n_variables, name='edge')
    visited_q = QuantumRegister(num_nodes, name='visited')
    incoming_a = AncillaRegister(num_nodes, name='incoming')
    zero_deg_a = AncillaRegister(num_nodes, name='zero_deg')
    acyclic_ok_a = AncillaRegister(1, name='acyclic_ok')
    
    # Temporary ancilla for intermediate calculations
    temp_a = AncillaRegister(1, name='temp')

    qc = QuantumCircuit(edge_q, visited_q, incoming_a, zero_deg_a, acyclic_ok_a, temp_a)

    # --- Kahn's Algorithm: n iterations for n nodes ---
    for _ in range(num_nodes):
        # --- Step 1: Calculate in-degrees for all nodes from unvisited sources ---
        for i in range(num_nodes):
            node_i_name = node_names[i]
            
            # For each potential incoming edge from j to i, calculate:
            # (edge[j->i] AND NOT visited[j])
            # and OR all results into incoming_a[i]
            for j in range(num_nodes):
                if i == j: continue
                node_j_name = node_names[j]
                
                if (node_j_name, node_i_name, 'direct') in causal_dict:
                    generic_var = causal_dict[(node_j_name, node_i_name, 'direct')]
                    if generic_var in reverse_cnf_map:
                        cnf_var = reverse_cnf_map[generic_var]
                        edge_qubit = edge_q[cnf_var - 1]
                        
                        # Compute (edge[j->i] AND NOT visited[j]) into temp_a
                        qc.x(visited_q[j])
                        qc.mcx([edge_qubit, visited_q[j]], temp_a[0])
                        
                        # Use the result to conditionally flip incoming_a[i]
                        # This creates an OR effect.
                        qc.cx(temp_a[0], incoming_a[i])
                        
                        # Uncompute the AND to clean the temp_a for the next use
                        qc.mcx([edge_qubit, visited_q[j]], temp_a[0])
                        qc.x(visited_q[j])
            
            # incoming_a[i] is now 1 if there's at least one incoming edge.
            # We flip it so it's 1 if there are ZERO incoming edges.
            qc.x(incoming_a[i])

        qc.barrier(label=f"Iter {_}")

        # --- Step 2 & 3: Find zero-in-degree nodes and mark as visited ---
        for i in range(num_nodes):
            # A node has zero-in-degree if it's NOT visited AND has NO incoming edges.
            qc.x(visited_q[i])
            qc.mcx([visited_q[i], incoming_a[i]], zero_deg_a[i])
            qc.x(visited_q[i])
            
            # "Remove" the node by marking it as visited
            qc.cx(zero_deg_a[i], visited_q[i])

        # --- Step 4: Uncompute this iteration to clean ancillas ---
        for i in range(num_nodes - 1, -1, -1):
            qc.cx(zero_deg_a[i], visited_q[i])
            qc.x(visited_q[i])
            qc.mcx([visited_q[i], incoming_a[i]], zero_deg_a[i])
            qc.x(visited_q[i])
            qc.x(incoming_a[i])

    # --- Final Check: Graph is acyclic if ALL nodes were visited ---
    qc.mcx(visited_q[:], acyclic_ok_a[0])
    
    return qc

def create_circuit(qc, n_variables, cnf, alpha, beta, qubit_map, oracles):
    """Builds one full Grover-style iteration."""
    edge_qubits = qubit_map['edge']
    final_ok_qubit = qubit_map['final_ok']
    cnf_oracle, acyclicity_oracle = oracles['cnf'], oracles['acyclic']
    
    qc.append(cnf_oracle, qubit_map['cnf_oracle_qubits'])
    qc.append(acyclicity_oracle, qubit_map['acyclic_oracle_qubits'])
    
    qc.mcx([qubit_map['cnf_ok'], qubit_map['acyclic_ok']], final_ok_qubit)
    qc.p(beta, final_ok_qubit)
    qc.mcx([qubit_map['cnf_ok'], qubit_map['acyclic_ok']], final_ok_qubit)
    
    qc.append(acyclicity_oracle.inverse(), qubit_map['acyclic_oracle_qubits'])
    qc.append(cnf_oracle.inverse(), qubit_map['cnf_oracle_qubits'])
    qc.barrier(label="Oracle")
    
    qc.h(edge_qubits)
    qc.x(edge_qubits)
    qc.h(edge_qubits[-1])
    qc.mcx(edge_qubits[:-1], edge_qubits[-1])
    qc.h(edge_qubits[-1])
    qc.x(edge_qubits)
    qc.h(edge_qubits)
    qc.barrier(label="Diffuser")


def solveFixedQuantunSAT(cnf, node_names, causal_dict, reverse_cnf_map, l_iterations, delta, debug=False, simulation=True):
    """Main entry point for the quantum solver."""
    variables = set(abs(var) for clause in cnf for var in clause)
    n_variables = len(variables)
    n_clauses = len(cnf)
    num_nodes = len(node_names)
    
    cnf_oracle = build_cnf_oracle(cnf, n_variables, n_clauses)
    acyclicity_oracle = build_acyclicity_oracle(num_nodes, node_names, causal_dict, reverse_cnf_map, n_variables)
    
    edge_q = QuantumRegister(n_variables, name='edge')
    clause_a = QuantumRegister(n_clauses, name='clause')
    cnf_ok_a = QuantumRegister(1, name='cnf_ok')
    visited_q = QuantumRegister(num_nodes, name='visited')
    incoming_a = QuantumRegister(num_nodes, name='incoming')
    zero_deg_a = QuantumRegister(num_nodes, name='zero_deg')
    acyclic_ok_a = QuantumRegister(1, name='acyclic_ok')
    temp_a = QuantumRegister(1, name='temp')
    final_ok_a = QuantumRegister(1, name='final_ok')
    
    # FIX: Add a ClassicalRegister for measurement results
    c_reg = ClassicalRegister(n_variables, name='c')
    
    # FIX: Include the ClassicalRegister in the circuit definition
    qc = QuantumCircuit(edge_q, clause_a, cnf_ok_a, visited_q, incoming_a, zero_deg_a, acyclic_ok_a, temp_a, final_ok_a, c_reg)
    
    qubit_map = {
        'edge': edge_q, 'final_ok': final_ok_a[0],
        'cnf_ok': cnf_ok_a[0], 'acyclic_ok': acyclic_ok_a[0],
        'cnf_oracle_qubits': edge_q[:] + clause_a[:] + cnf_ok_a[:],
        'acyclic_oracle_qubits': edge_q[:] + visited_q[:] + incoming_a[:] + zero_deg_a[:] + acyclic_ok_a[:] + temp_a[:]
    }
    
    print(f"LOG: Circuit requires {qc.num_qubits} qubits ({n_variables} for solution, {qc.num_qubits - n_variables} ancillas).")
    
    if qc.num_qubits > 28 and simulation:
        print("LOG: Too many qubits for efficient simulation, aborting.")
        return False, []
    
    qc.h(edge_q)
    
    L = 2 * l_iterations + 1
    gamma_inv = chebyshev(1/L, 1/delta)
    alpha_values = np.array([2*mpm.acot(mpm.tan(2*mpm.pi*(i+1)/L) * mpm.sqrt(1-1/gamma_inv**2)) for i in range(l_iterations)], dtype=complex).real
    beta_values = -alpha_values[::-1]
    
    oracles = {'cnf': cnf_oracle, 'acyclic': acyclicity_oracle}

    for i in range(l_iterations):
        create_circuit(qc, n_variables, cnf, alpha_values[i], beta_values[i], qubit_map, oracles)
    
    print(f"LOG: Circuit created, depth: {qc.depth()}")
    
    # FIX: Measure the edge qubits into the classical register
    qc.measure(edge_q, c_reg)
    
    if debug:
        circuit_drawer(qc, output="mpl")
        plt.savefig("debug/fixed-circuit.png")
        plt.close()
    
    if debug:
        cnf_oracle.draw('mpl').savefig('debug/cnf_oracle.png')
        acyclicity_oracle.draw('mpl').savefig('debug/acyclicity_oracle.png')
        
    qc = RemoveBarriers()(qc)
    optimized_qc = transpile(qc, optimization_level=3)

    if simulation:
        result = Sampler().run([optimized_qc], shots=1024).result()
        counts = result.quasi_dists[0]
        counts = counts.binary_probabilities(num_bits=qc.num_qubits)
    else:
        print("LOG: Real hardware execution not implemented.")
        return False, []
    
    if debug: print(f"DEBUG: clustering solutions, {len(counts)}")
    temp_counts, sil = cluster_solutions(counts)
    
    
    
    print(f"LOG: Silhouette score: {sil}")
    
    if debug: print(f"DEBUG: clustered solutions, {len(counts)}")
    
    if debug: elbow_plot(counts, temp_counts)
    
    
    # doing this just to debug the clustering, sorry for the mess
    # counts = temp_counts
    
    if debug: 
        plot_histogram(counts)
        plt.savefig('debug/fixed-histogram.png')
        plt.close()
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
