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
    qc = QuantumCircuit(edge_q, clause_a, cnf_ok_a, name="CNF_Oracle")

    # For each clause, check if it's UNSATISFIED
    # A clause is unsatisfied if all its literals are false.
    # (e.g., for clause (x1 V !x2), it's unsatisfied if x1=0 and x2=1)
    for i, clause in enumerate(cnf):
        # Flip literals to check for the unsatisfied condition
        for literal in clause:
            if literal > 0: # Positive literal (x_i) is false when qubit is |0>
                qc.x(edge_q[abs(literal) - 1])
        
        # Get control qubits for the current clause
        control_indices = [edge_q[abs(literal) - 1] for literal in clause]
        
        # If all literals are in their "false" state, flip the clause ancilla
        qc.mcx(control_indices, clause_a[i])

        # Unflip the literals to restore state
        for literal in clause:
            if literal > 0:
                qc.x(edge_q[abs(literal) - 1])
    
    # The CNF is satisfied if NO clause ancilla was flipped to |1>.
    # So, we check if all clause ancillas are |0>.
    qc.x(clause_a) # Flip all clause ancillas
    qc.mcx(clause_a[:], cnf_ok_a[0]) # If all are |1> (originally |0>), flip cnf_ok
    qc.x(clause_a) # Uncompute
    
    return qc

def build_acyclicity_oracle(num_nodes, node_names, causal_dict, reverse_cnf_map, n_variables):
    """
    Builds the acyclicity oracle based on Kahn's algorithm.
    The graph is acyclic if the algorithm can visit/remove every node.
    """
    # --- Define Quantum Registers ---
    edge_q = QuantumRegister(n_variables, name='edge')
    visited_q = QuantumRegister(num_nodes, name='visited') # Tracks visited nodes
    
    # --- Ancilla Registers ---
    # incoming_a[i] will be 1 if node i has ZERO in-degree from unvisited nodes
    incoming_a = AncillaRegister(num_nodes, name='incoming') 
    # zero_deg_a[i] is a helper to mark nodes for visiting in the current iteration
    zero_deg_a = AncillaRegister(num_nodes, name='zero_deg') 
    acyclic_ok_a = AncillaRegister(1, name='acyclic_ok')
    
    qc = QuantumCircuit(edge_q, visited_q, incoming_a, zero_deg_a, acyclic_ok_a, name="Acyclicity_Oracle")

    # --- Kahn's Algorithm: n iterations for n nodes ---
    for iter_num in range(num_nodes):
        # --- COMPUTE PHASE ---
        
        # Step 1: Calculate in-degrees for all nodes from unvisited sources.
        # We initialize `incoming_a` to all |1>s, assuming zero in-degree.
        # We then flip to |0> if we find any incoming edge from an unvisited node.
        qc.x(incoming_a)
        
        for i in range(num_nodes): # Target node i
            for j in range(num_nodes): # Source node j
                if i == j: continue
                
                node_i_name = node_names[i]
                node_j_name = node_names[j]

                # Check if a potential edge j->i exists in our variable mapping
                if (node_j_name, node_i_name, 'direct') in causal_dict:
                    generic_var = causal_dict[(node_j_name, node_i_name, 'direct')]
                    if generic_var in reverse_cnf_map:
                        cnf_var = reverse_cnf_map[generic_var]
                        edge_qubit = edge_q[cnf_var - 1]
                        
                        # Condition: edge j->i exists AND node j is NOT visited.
                        # If this is true, flip incoming_a[i] from 1 to 0.
                        qc.x(visited_q[j])
                        qc.ccx(edge_qubit, visited_q[j], incoming_a[i])
                        qc.x(visited_q[j]) # Uncompute

        # Step 2: Identify nodes to "remove" in this iteration.
        # A node can be removed if it has not been visited AND has zero in-degree.
        for i in range(num_nodes):
            # zero_deg_a[i] = (NOT visited_q[i]) AND (incoming_a[i])
            qc.x(visited_q[i])
            qc.ccx(visited_q[i], incoming_a[i], zero_deg_a[i])
            qc.x(visited_q[i])
            
            # Step 3: Mark the identified nodes as visited.
            # visited_q[i] |= zero_deg_a[i]
            qc.cx(zero_deg_a[i], visited_q[i])

        qc.barrier(label=f"Iter {iter_num}")

        # --- UNCOMPUTE PHASE ---
        # To keep the circuit reversible and clean the ancillas for the next iteration,
        # we must reverse the computations of this iteration.
        # The state of `visited_q` is the only thing that should persist.

        # Reverse Step 2 & 3
        for i in range(num_nodes - 1, -1, -1):
            qc.cx(zero_deg_a[i], visited_q[i]) # Must be undone first
            qc.x(visited_q[i])
            qc.ccx(visited_q[i], incoming_a[i], zero_deg_a[i]) # Uncomputes zero_deg_a
            qc.x(visited_q[i])

        # Reverse Step 1
        for i in range(num_nodes - 1, -1, -1):
            for j in range(num_nodes - 1, -1, -1):
                if i == j: continue
                node_i_name = node_names[i]
                node_j_name = node_names[j]
                if (node_j_name, node_i_name, 'direct') in causal_dict:
                    generic_var = causal_dict[(node_j_name, node_i_name, 'direct')]
                    if generic_var in reverse_cnf_map:
                        cnf_var = reverse_cnf_map[generic_var]
                        edge_qubit = edge_q[cnf_var - 1]
                        qc.x(visited_q[j])
                        qc.ccx(edge_qubit, visited_q[j], incoming_a[i])
                        qc.x(visited_q[j])
        qc.x(incoming_a) # Uncompute initialization

    # --- Final Check ---
    # The graph is acyclic if ALL nodes were visited.
    qc.mcx(visited_q[:], acyclic_ok_a[0])
    
    return qc

def create_circuit(qc, alpha, beta, qubit_map, oracles):
    """Builds one full Grover-style iteration."""
    # --- Oracle Phase ---
    # Apply both oracles to mark solutions where CNF is SAT and graph is ACYCLIC.
    qc.append(oracles['cnf'], qubit_map['cnf_oracle_qubits'])
    # FIX: Acyclicity oracle was commented out, now it is enabled.
    qc.append(oracles['acyclic'], qubit_map['acyclic_oracle_qubits'])
    
    # Mark states where both conditions are met
    qc.mcx([qubit_map['cnf_ok'], qubit_map['acyclic_ok']], qubit_map['final_ok'])
    
    # Apply phase shift (part of a specific amplitude amplification variant)
    qc.p(beta, qubit_map['final_ok'])
    
    # Uncompute to restore the final_ok ancilla
    qc.mcx([qubit_map['cnf_ok'], qubit_map['acyclic_ok']], qubit_map['final_ok'])
    
    # Uncompute the oracles
    # FIX: Acyclicity oracle inverse was commented out, now it is enabled.
    qc.append(oracles['acyclic'].inverse(), qubit_map['acyclic_oracle_qubits'])
    qc.append(oracles['cnf'].inverse(), qubit_map['cnf_oracle_qubits'])
    qc.barrier(label="Oracle")
    
    # --- Diffuser Phase ---
    edge_qubits = qubit_map['edge']
    # qc.h(edge_qubits)
    # qc.x(edge_qubits)
    # qc.h(edge_qubits[-1])
    # qc.mcx(edge_qubits[:-1], edge_qubits[-1])
    # qc.h(edge_qubits[-1])
    # qc.x(edge_qubits)
    # qc.h(edge_qubits)
    # qc.barrier(label="Diffuser")
    
    qc.h(edge_qubits)
    qc.barrier()
    qc.x(edge_qubits[:-1])
    qc.barrier()

    qc.p(-alpha/2, edge_qubits[-1])
    qc.barrier()
    qc.mcx(edge_qubits[:-1], edge_qubits[-1])
    qc.mcx(edge_qubits[:-1], qubit_map['final_ok'])
    qc.barrier()

    qc.p(-alpha/2, edge_qubits[-1])
    qc.p(-alpha/2, qubit_map['final_ok'])
    qc.barrier()
    qc.mcx(edge_qubits[:-1], edge_qubits[-1])
    qc.mcx(edge_qubits[:-1], qubit_map['final_ok'])

    qc.p(alpha, edge_qubits[-1])
    
    qc.barrier()
    qc.x(edge_qubits[:-1])
    qc.barrier()
    qc.h(edge_qubits)
    qc.barrier()



def solveFixedQuantumSAT(cnf, node_names, causal_dict, reverse_cnf_map, l_iterations, delta, debug=False, simulation=True):
    """Main entry point for the quantum solver."""
    all_vars_in_cnf = {abs(var) for clause in cnf for var in clause}
    n_variables = max(all_vars_in_cnf) if all_vars_in_cnf else 0
    
    n_clauses = len(cnf)
    num_nodes = len(node_names)
    
    # --- Build Oracle Circuits ---
    cnf_oracle = build_cnf_oracle(cnf, n_variables, n_clauses)
    acyclicity_oracle = build_acyclicity_oracle(num_nodes, node_names, causal_dict, reverse_cnf_map, n_variables)
    
    # --- Define Main Circuit Registers ---
    edge_q = QuantumRegister(n_variables, name='edge')
    clause_a = QuantumRegister(n_clauses, name='clause')
    cnf_ok_a = QuantumRegister(1, name='cnf_ok')
    visited_q = QuantumRegister(num_nodes, name='visited')
    incoming_a = QuantumRegister(num_nodes, name='incoming')
    zero_deg_a = QuantumRegister(num_nodes, name='zero_deg')
    acyclic_ok_a = QuantumRegister(1, name='acyclic_ok')
    final_ok_a = QuantumRegister(1, name='final_ok')
    
    c_reg = ClassicalRegister(n_variables, name='c')
    
    qc = QuantumCircuit(edge_q, clause_a, cnf_ok_a, visited_q, incoming_a, zero_deg_a, acyclic_ok_a, final_ok_a, c_reg)
    
    # --- Map registers to oracle inputs ---
    qubit_map = {
        'edge': edge_q, 'final_ok': final_ok_a[0],
        'cnf_ok': cnf_ok_a[0], 'acyclic_ok': acyclic_ok_a[0],
        'cnf_oracle_qubits': edge_q[:] + clause_a[:] + cnf_ok_a[:],
        # FIX: Adjusted map for the new acyclicity oracle's registers
        'acyclic_oracle_qubits': edge_q[:] + visited_q[:] + incoming_a[:] + zero_deg_a[:] + acyclic_ok_a[:]
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
        create_circuit(qc, alpha_values[i], beta_values[i], qubit_map, oracles)
    
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
            # The bitstring from qiskit is often little-endian, so we reverse it.
            bitstring = bitstring[::-1]
            solution = []
            for i in range(n_variables):
                var_num = i + 1
                if bitstring[i] == '0':
                    solution.append(-var_num)
                else:
                    solution.append(var_num)
            solutions.append(solution)
    
    # print(f"DEBUG: solutions={solutions}")
        
    return is_sat, solutions
