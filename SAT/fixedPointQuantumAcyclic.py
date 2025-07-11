import numpy as np
import mpmath as mpm
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_ibm_runtime import SamplerV2
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from utils import structural_check, elbow_plot, cluster_solutions

def chebyshev(L, x):
    """Calculates the Chebyshev polynomial of the first kind."""
    return mpm.cos(L * mpm.acos(x))

def build_cnf_oracle(cnf, n_variables, n_clauses):
    """Builds the CNF checking logic as a separate, reversible circuit."""
    
    print(cnf, n_variables, n_clauses)
    
    edge_q = QuantumRegister(n_variables, name='edge')
    clause_a = AncillaRegister(n_clauses, name='clause')
    cnf_ok_a = AncillaRegister(1, name='cnf_ok')
    qc = QuantumCircuit(edge_q, clause_a, cnf_ok_a, name="CNF_Oracle")

    for i, clause in enumerate(cnf):
        for literal in clause:
            if literal > 0:
                qc.x(edge_q[abs(literal) - 1])
        
        control_indices = [edge_q[abs(literal) - 1] for literal in clause]
        qc.mcx(control_indices, clause_a[i])
        for literal in clause:
            if literal > 0:
                qc.x(edge_q[abs(literal) - 1])
    
    qc.x(clause_a)
    qc.mcx(clause_a[:], cnf_ok_a[0])
    qc.x(clause_a)
    
    return qc

def build_acyclicity_oracle(num_nodes, node_names, causal_dict, causal_to_cnf_map, n_variables):
    """
    Builds a reversible acyclicity oracle using Kahn's algorithm with proper uncomputation.
    The key insight is to store the computation history in additional ancillas.
    """
    edge_q = QuantumRegister(n_variables, name='edge')
    
    # For each iteration, we need to store:
    # 1. The state of visited nodes at that iteration
    # 2. Which nodes had zero in-degree at that iteration
    # 3. Which nodes were processed in that iteration
    
    # History ancillas - one set for each iteration
    visited_history = []
    zero_indegree_history = []
    processed_history = []
    
    for iteration in range(num_nodes):
        visited_history.append(AncillaRegister(num_nodes, name=f'visited_hist_{iteration}'))
        zero_indegree_history.append(AncillaRegister(num_nodes, name=f'zero_indeg_hist_{iteration}'))
        processed_history.append(AncillaRegister(num_nodes, name=f'processed_hist_{iteration}'))
    
    # Working registers
    visited_q = QuantumRegister(num_nodes, name='visited')
    zero_indegree_q = QuantumRegister(num_nodes, name='zero_indegree')
    temp_ancillas = AncillaRegister(num_nodes * num_nodes, name='temp')
    acyclic_ok_a = AncillaRegister(1, name='acyclic_ok')
    
    # Build the circuit
    all_registers = [edge_q, visited_q, zero_indegree_q, temp_ancillas, acyclic_ok_a]
    for i in range(num_nodes):
        all_registers.extend([visited_history[i], zero_indegree_history[i], processed_history[i]])
    
    qc = QuantumCircuit(*all_registers, name="Acyclicity_Oracle_Fixed_Kahns")
    
    # Build edge mapping
    edge_map = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                key = (node_names[i], node_names[j], 'direct')
                if key in causal_dict and causal_dict[key] in causal_to_cnf_map:
                    cnf_var = causal_to_cnf_map[causal_dict[key]]
                    edge_map[(i, j)] = cnf_var - 1
    
    # Kahn's algorithm iterations
    for iteration in range(num_nodes):
        qc.barrier(label=f"Iteration_{iteration}_Start")
        
        # Save current state to history
        for node_idx in range(num_nodes):
            qc.cx(visited_q[node_idx], visited_history[iteration][node_idx])
        
        # Compute zero in-degree nodes
        for node_idx in range(num_nodes):
            # Count incoming edges from unvisited nodes
            incoming_controls = []
            for k in range(num_nodes):
                if k != node_idx and (k, node_idx) in edge_map:
                    temp_idx = k * num_nodes + node_idx
                    if temp_idx < len(temp_ancillas):
                        # temp = edge_exists AND node_k_not_visited
                        qc.x(visited_q[k])  # Flip to get "not visited"
                        qc.ccx(edge_q[edge_map[(k, node_idx)]], visited_q[k], temp_ancillas[temp_idx])
                        qc.x(visited_q[k])  # Flip back
                        incoming_controls.append(temp_ancillas[temp_idx])
            
            # Zero in-degree if no incoming edges from unvisited nodes AND node is unvisited
            if incoming_controls:
                # Use De Morgan's law: NOT(A OR B OR C) = NOT A AND NOT B AND NOT C
                for ctrl in incoming_controls:
                    qc.x(ctrl)
                qc.x(visited_q[node_idx])  # Flip to get "not visited"
                qc.mcx(incoming_controls + [visited_q[node_idx]], zero_indegree_q[node_idx])
                qc.x(visited_q[node_idx])  # Flip back
                for ctrl in incoming_controls:
                    qc.x(ctrl)
            else:
                # No incoming edges, so zero in-degree if unvisited
                qc.x(visited_q[node_idx])
                qc.cx(visited_q[node_idx], zero_indegree_q[node_idx])
                qc.x(visited_q[node_idx])
        
        # Save zero in-degree state to history
        for node_idx in range(num_nodes):
            qc.cx(zero_indegree_q[node_idx], zero_indegree_history[iteration][node_idx])
        
        # Process nodes with zero in-degree (mark as visited)
        for node_idx in range(num_nodes):
            qc.cx(zero_indegree_q[node_idx], processed_history[iteration][node_idx])
            qc.cx(zero_indegree_q[node_idx], visited_q[node_idx])
        
        # Uncompute zero in-degree computation (using history)
        for node_idx in range(num_nodes):
            incoming_controls = []
            for k in range(num_nodes):
                if k != node_idx and (k, node_idx) in edge_map:
                    temp_idx = k * num_nodes + node_idx
                    if temp_idx < len(temp_ancillas):
                        incoming_controls.append(temp_ancillas[temp_idx])
            
            if incoming_controls:
                for ctrl in incoming_controls:
                    qc.x(ctrl)
                qc.x(visited_history[iteration][node_idx])  # Use history state
                qc.mcx(incoming_controls + [visited_history[iteration][node_idx]], zero_indegree_q[node_idx])
                qc.x(visited_history[iteration][node_idx])
                for ctrl in incoming_controls:
                    qc.x(ctrl)
            else:
                qc.x(visited_history[iteration][node_idx])
                qc.cx(visited_history[iteration][node_idx], zero_indegree_q[node_idx])
                qc.x(visited_history[iteration][node_idx])
        
        # Uncompute temp ancillas
        for node_idx in range(num_nodes):
            for k in range(num_nodes):
                if k != node_idx and (k, node_idx) in edge_map:
                    temp_idx = k * num_nodes + node_idx
                    if temp_idx < len(temp_ancillas):
                        qc.x(visited_history[iteration][k])
                        qc.ccx(edge_q[edge_map[(k, node_idx)]], visited_history[iteration][k], temp_ancillas[temp_idx])
                        qc.x(visited_history[iteration][k])
    
    qc.barrier(label="Final_Check")
    
    # Final acyclicity check: all nodes should be visited
    qc.mcx(visited_q[:], acyclic_ok_a[0])
    
    qc.barrier(label="Uncomputation_Start")
    
    # Uncompute everything in reverse order using the stored history
    for iteration in range(num_nodes - 1, -1, -1):
        qc.barrier(label=f"Uncompute_Iteration_{iteration}")
        
        # Recompute temp ancillas for this iteration
        for node_idx in range(num_nodes):
            for k in range(num_nodes):
                if k != node_idx and (k, node_idx) in edge_map:
                    temp_idx = k * num_nodes + node_idx
                    if temp_idx < len(temp_ancillas):
                        qc.x(visited_history[iteration][k])
                        qc.ccx(edge_q[edge_map[(k, node_idx)]], visited_history[iteration][k], temp_ancillas[temp_idx])
                        qc.x(visited_history[iteration][k])
        
        # Recompute zero in-degree using history
        for node_idx in range(num_nodes):
            incoming_controls = []
            for k in range(num_nodes):
                if k != node_idx and (k, node_idx) in edge_map:
                    temp_idx = k * num_nodes + node_idx
                    if temp_idx < len(temp_ancillas):
                        incoming_controls.append(temp_ancillas[temp_idx])
            
            if incoming_controls:
                for ctrl in incoming_controls:
                    qc.x(ctrl)
                qc.x(visited_history[iteration][node_idx])
                qc.mcx(incoming_controls + [visited_history[iteration][node_idx]], zero_indegree_q[node_idx])
                qc.x(visited_history[iteration][node_idx])
                for ctrl in incoming_controls:
                    qc.x(ctrl)
            else:
                qc.x(visited_history[iteration][node_idx])
                qc.cx(visited_history[iteration][node_idx], zero_indegree_q[node_idx])
                qc.x(visited_history[iteration][node_idx])
        
        # Uncompute the processing (reverse the visited updates)
        for node_idx in range(num_nodes):
            qc.cx(zero_indegree_q[node_idx], visited_q[node_idx])
            qc.cx(zero_indegree_q[node_idx], processed_history[iteration][node_idx])
        
        # Uncompute zero in-degree history
        for node_idx in range(num_nodes):
            qc.cx(zero_indegree_q[node_idx], zero_indegree_history[iteration][node_idx])
        
        # Uncompute zero in-degree computation
        for node_idx in range(num_nodes):
            incoming_controls = []
            for k in range(num_nodes):
                if k != node_idx and (k, node_idx) in edge_map:
                    temp_idx = k * num_nodes + node_idx
                    if temp_idx < len(temp_ancillas):
                        incoming_controls.append(temp_ancillas[temp_idx])
            
            if incoming_controls:
                for ctrl in incoming_controls:
                    qc.x(ctrl)
                qc.x(visited_history[iteration][node_idx])
                qc.mcx(incoming_controls + [visited_history[iteration][node_idx]], zero_indegree_q[node_idx])
                qc.x(visited_history[iteration][node_idx])
                for ctrl in incoming_controls:
                    qc.x(ctrl)
            else:
                qc.x(visited_history[iteration][node_idx])
                qc.cx(visited_history[iteration][node_idx], zero_indegree_q[node_idx])
                qc.x(visited_history[iteration][node_idx])
        
        # Uncompute temp ancillas
        for node_idx in range(num_nodes):
            for k in range(num_nodes):
                if k != node_idx and (k, node_idx) in edge_map:
                    temp_idx = k * num_nodes + node_idx
                    if temp_idx < len(temp_ancillas):
                        qc.x(visited_history[iteration][k])
                        qc.ccx(edge_q[edge_map[(k, node_idx)]], visited_history[iteration][k], temp_ancillas[temp_idx])
                        qc.x(visited_history[iteration][k])
        
        # Uncompute visited history
        for node_idx in range(num_nodes):
            qc.cx(visited_q[node_idx], visited_history[iteration][node_idx])
    
    return qc

def create_circuit(qc, alpha, beta, qubit_map, oracles):
    """Builds one full Grover-style iteration."""
    # --- Oracle Phase ---
    qc.append(oracles['cnf'], qubit_map['cnf_oracle_qubits'])
    qc.append(oracles['acyclic'], qubit_map['acyclic_oracle_qubits'])
    
    qc.mcx([qubit_map['cnf_ok'], qubit_map['acyclic_ok']], qubit_map['final_ok'])
    qc.p(beta, qubit_map['final_ok'])
    qc.mcx([qubit_map['cnf_ok'], qubit_map['acyclic_ok']], qubit_map['final_ok'])
    
    qc.append(oracles['acyclic'].inverse(), qubit_map['acyclic_oracle_qubits'])
    qc.append(oracles['cnf'].inverse(), qubit_map['cnf_oracle_qubits'])
    qc.barrier(label="Oracle")
    
    # qc.append(oracles['cnf'], qubit_map['cnf_oracle_qubits'])
    
    # qc.mcx([qubit_map['cnf_ok']], qubit_map['final_ok'])
    # qc.p(beta, qubit_map['final_ok'])
    # qc.mcx([qubit_map['cnf_ok']], qubit_map['final_ok'])
    
    # qc.append(oracles['cnf'].inverse(), qubit_map['cnf_oracle_qubits'])
    # qc.barrier(label="Oracle")
    
    # --- Diffuser Phase (Standard Grover Diffuser) ---
    edge_qubits = qubit_map['edge']
    qc.h(edge_qubits)
    qc.x(edge_qubits[:-1])
    qc.p(-alpha/2, edge_qubits[-1])
    qc.mcx(edge_qubits[:-1], edge_qubits[-1])
    qc.mcx(edge_qubits[:-1], qubit_map['final_ok'])
    qc.p(-alpha/2, edge_qubits[-1])
    qc.p(-alpha/2, qubit_map['final_ok']) 
    qc.mcx(edge_qubits[:-1], edge_qubits[-1])
    qc.mcx(edge_qubits[:-1], qubit_map['final_ok'])
    qc.p(alpha, edge_qubits[-1])
    qc.x(edge_qubits[:-1])
    qc.h(edge_qubits)
    qc.barrier(label="Diffuser")

def solveFixedQuantumSAT(cnf, node_names, causal_dict, causal_to_cnf_map, l_iterations, delta, debug=False, simulation=True):
    """Main entry point for the quantum solver."""
    all_vars_in_cnf = {abs(var) for clause in cnf for var in clause}
    n_variables = max(all_vars_in_cnf) if all_vars_in_cnf else 0
    n_clauses = len(cnf)
    num_nodes = len(node_names)
    
    # Create oracles
    cnf_oracle = build_cnf_oracle(cnf, n_variables, n_clauses)
    acyclicity_oracle = build_acyclicity_oracle(num_nodes, node_names, causal_dict, causal_to_cnf_map, n_variables)
    
    # Initialize registers for the main circuit
    edge_q = QuantumRegister(n_variables, name='edge')
    clause_a = QuantumRegister(n_clauses, name='clause')
    cnf_ok_a = QuantumRegister(1, name='cnf_ok')
    
    # Create registers for Kahn's algorithm acyclicity oracle
    visited_q = QuantumRegister(num_nodes, name='visited')
    zero_indegree_q = QuantumRegister(num_nodes, name='zero_indegree')
    temp_ancillas = AncillaRegister(num_nodes * num_nodes, name='temp')
    acyclic_ok_a = AncillaRegister(1, name='acyclic_ok')
    
    # Create history registers for Kahn's algorithm
    visited_history = []
    zero_indegree_history = []
    processed_history = []
    
    for iteration in range(num_nodes):
        visited_history.append(AncillaRegister(num_nodes, name=f'visited_hist_{iteration}'))
        zero_indegree_history.append(AncillaRegister(num_nodes, name=f'zero_indeg_hist_{iteration}'))
        processed_history.append(AncillaRegister(num_nodes, name=f'processed_hist_{iteration}'))
    
    final_ok_a = AncillaRegister(1, name='final_ok')
    c_reg = ClassicalRegister(n_variables, name='c')

    # Create the main quantum circuit with all registers in the same order as the oracle
    all_registers = [edge_q, clause_a, cnf_ok_a, visited_q, zero_indegree_q, 
                     temp_ancillas, acyclic_ok_a, final_ok_a]
    
    # Add history registers in the same order as in the oracle
    for i in range(num_nodes):
        all_registers.append(visited_history[i])
        all_registers.append(zero_indegree_history[i])
        all_registers.append(processed_history[i])
    
    all_registers.append(c_reg)  # Add classical register last
    qc = QuantumCircuit(*all_registers)
    
    # Create qubit map that exactly matches the acyclicity oracle's structure
    acyclic_oracle_qubits = [edge_q[:], visited_q[:], zero_indegree_q[:], temp_ancillas[:], acyclic_ok_a[:]]
    for i in range(num_nodes):
        acyclic_oracle_qubits.append(visited_history[i][:])
        acyclic_oracle_qubits.append(zero_indegree_history[i][:])
        acyclic_oracle_qubits.append(processed_history[i][:])
    
    # Flatten the list of qubit lists
    flat_acyclic_oracle_qubits = []
    for qubits in acyclic_oracle_qubits:
        flat_acyclic_oracle_qubits.extend(qubits)
    
    qubit_map = {
        'edge': edge_q, 
        'final_ok': final_ok_a[0],
        'cnf_ok': cnf_ok_a[0], 
        'acyclic_ok': acyclic_ok_a[0],
        'cnf_oracle_qubits': edge_q[:] + clause_a[:] + cnf_ok_a[:],
        'acyclic_oracle_qubits': flat_acyclic_oracle_qubits
    }
    
    print(f"LOG: Circuit requires {qc.num_qubits} qubits ({n_variables} for solution, {qc.num_qubits - n_variables} ancillas).")
    print(f"LOG: Acyclicity oracle expects {len(flat_acyclic_oracle_qubits)} qubits")
    
    # if qc.num_qubits > 28 and simulation:
    #     print("LOG: Too many qubits for efficient simulation, aborting.")
    #     return False, []
    
    # Initialize qubits to superposition
    qc.h(edge_q)
    
    L = 2 * l_iterations + 1
    gamma_inv = chebyshev(1/L, 1/delta)
    alpha_values = np.array([2*mpm.acot(mpm.tan(2*mpm.pi*(i+1)/L) * mpm.sqrt(1-1/gamma_inv**2)) for i in range(l_iterations)], dtype=complex).real
    beta_values = -alpha_values[::-1]
    
    oracles = {'cnf': cnf_oracle, 'acyclic': acyclicity_oracle}

    for i in range(l_iterations):
        create_circuit(qc, alpha_values[i], beta_values[i], qubit_map, oracles)
    
    print(f"LOG: Circuit created, depth: {qc.depth()}")
    
    qc.measure(edge_q, c_reg)
    
    # if qc.num_qubits > 28 and simulation:
    #     print("LOG: Too many qubits for efficient simulation, aborting.")
    #     return False, []
    
    # ... (rest of the function is the same as before) ...
    if debug:
        print("DEBUG: Saving circuit diagrams...")
        circuit_drawer(qc, output="mpl", fold=-1).savefig("debug/fixed-circuit-acyclic.png")
        plt.close()
        cnf_oracle.draw('mpl', fold=-1).savefig('debug/cnf_oracle.png')
        plt.close()
        acyclicity_oracle.draw('mpl', fold=-1).savefig('debug/acyclicity_oracle.png')
        plt.close()
        
    qc = RemoveBarriers()(qc)
    optimized_qc = qc

    if simulation:
        result = Sampler().run([optimized_qc], shots=1024).result()
        counts = result.quasi_dists[0].binary_probabilities(num_bits=n_variables)
    else:
        print("LOG: Real hardware execution not implemented.")
        return False, []
    
    if debug: print(f"DEBUG: Clustering {len(counts)} solutions.")
    temp_counts, sil = cluster_solutions(counts)
    print(f"LOG: Silhouette score: {sil}")
    
    if debug: 
        elbow_plot(counts, temp_counts)
        plot_histogram(counts)
        plt.savefig('debug/fixed-histogram.png')
        plt.close()

    solutions = []
    if not counts:
        return False, []
    else:
        is_sat = True
        for bitstring, prob in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            bitstring = bitstring[::-1]
            solution = []
            for i in range(n_variables):
                var_num = i + 1
            
                if bitstring[i] == '0':
                    solution.append(-var_num)
                else:
                    solution.append(var_num)
            solutions.append(solution)
            
    return is_sat, solutions
