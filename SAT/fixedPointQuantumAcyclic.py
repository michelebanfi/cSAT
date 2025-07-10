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
    Builds a reversible acyclicity oracle.
    This version attempts to strictly uncompute all temporary ancillas.
    """
    edge_q = QuantumRegister(n_variables, name='edge')
    visited_q = QuantumRegister(num_nodes, name='visited') 
    # ancilla_zero_indegree: For each node, tracks if it has zero in-degree from unvisited nodes
    ancilla_zero_indegree = AncillaRegister(num_nodes, name='anc_zero_indeg')
    # ancilla_incoming_edge_temp: Temporary storage for each (edge_exists AND source_unvisited)
    ancilla_incoming_edge_temp = AncillaRegister(num_nodes, name='anc_incoming_temp') # One for each possible incoming edge
    # ancilla_or_temp: A single ancilla for the OR aggregation logic
    ancilla_or_temp = AncillaRegister(1, name='anc_or_temp') # Used for computing OR of incoming_edge_temp
    
    acyclic_ok_a = AncillaRegister(1, name='acyclic_ok')
    
    qc = QuantumCircuit(edge_q, visited_q, ancilla_zero_indegree, 
                        ancilla_incoming_edge_temp, ancilla_or_temp, # New registers
                        acyclic_ok_a, name="Acyclicity_Oracle")
    
    # Build edge mapping for efficient lookup
    edge_map = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                key = (node_names[i], node_names[j], 'direct')
                if key in causal_dict and causal_dict[key] in causal_to_cnf_map:
                    cnf_var = causal_to_cnf_map[causal_dict[key]]
                    edge_map[(i, j)] = cnf_var - 1  # Convert to 0-indexed qubit index

    # Iterate num_nodes times, simulating steps of Kahn's algorithm
    for iteration in range(num_nodes):
        qc.barrier(label=f"Kahn_Iter_{iteration}_Compute_InDegree")

        # Step 1: Compute which unvisited nodes have zero in-degree
        for node_idx in range(num_nodes):
            # Compute (edge_k_to_node_idx AND NOT visited_q[k]) into ancilla_incoming_edge_temp[k]
            # for all k.
            
            # This loop *must* uncompute its effect on ancilla_incoming_edge_temp[k]
            # at the end of computing ancilla_zero_indegree[node_idx]
            
            # 1a. Compute individual (edge AND NOT visited) flags
            # Use ancilla_incoming_edge_temp[k] to store: edge(k->node_idx) exists AND k is unvisited
            for k in range(num_nodes):
                if k != node_idx and (k, node_idx) in edge_map:
                    edge_qubit = edge_q[edge_map[(k, node_idx)]]
                    qc.x(visited_q[k]) # Invert visited_q[k] so it's 1 if unvisited
                    qc.ccx(edge_qubit, visited_q[k], ancilla_incoming_edge_temp[k])
                    qc.x(visited_q[k]) # Uninvert visited_q[k]

            # 1b. Aggregate to check if ANY incoming edge from unvisited exists
            # We want ancilla_or_temp[0] to be 1 if ANY ancilla_incoming_edge_temp[k] is 1
            # Implemented as NOT(AND(NOT A, NOT B, ...))
            
            controls_for_or_temp = []
            for k in range(num_nodes):
                 if k != node_idx and (k, node_idx) in edge_map: # Only include relevant ones
                    controls_for_or_temp.append(ancilla_incoming_edge_temp[k])
            
            if controls_for_or_temp:
                for ctrl in controls_for_or_temp: # Flip controls to 0 state
                    qc.x(ctrl)
                qc.mcx(controls_for_or_temp, ancilla_or_temp[0]) # If all controls are 0, target flips.
                for ctrl in controls_for_or_temp: # Unflip controls
                    qc.x(ctrl)
                qc.x(ancilla_or_temp[0]) # Flip target to make it an OR (1 if any input was 1)
            # If controls_for_or_temp is empty, ancilla_or_temp[0] should remain 0 (no incoming edges from unvisited)

            # 1c. Set ancilla_zero_indegree[node_idx] based on (NOT ancilla_or_temp[0]) AND (NOT visited_q[node_idx])
            # ancilla_zero_indegree[node_idx] should be 1 if node has zero in-degree AND is unvisited.
            # So, (NOT ancilla_or_temp[0]) means no incoming edges from unvisited
            # AND (NOT visited_q[node_idx]) means the node itself is unvisited.

            qc.x(ancilla_or_temp[0]) # ancilla_or_temp[0] now 1 if no incoming, 0 if incoming
            qc.x(visited_q[node_idx]) # visited_q[node_idx] now 1 if unvisited, 0 if visited
            qc.ccx(ancilla_or_temp[0], visited_q[node_idx], ancilla_zero_indegree[node_idx])
            qc.x(visited_q[node_idx]) # Unflip visited_q[node_idx]

            # 1d. Uncompute ancilla_or_temp[0] and ancilla_incoming_edge_temp
            # Reverse 1c
            qc.x(visited_q[node_idx]) # Flip visited_q for uncomputation
            qc.ccx(ancilla_or_temp[0], visited_q[node_idx], ancilla_zero_indegree[node_idx])
            qc.x(visited_q[node_idx]) # Unflip visited_q

            # Reverse 1b
            qc.x(ancilla_or_temp[0]) # Unflip OR target
            if controls_for_or_temp:
                for ctrl in reversed(controls_for_or_temp): # Unflip controls (reverse order)
                    qc.x(ctrl)
                qc.mcx(controls_for_or_temp, ancilla_or_temp[0]) # Uncompute NAND
                for ctrl in reversed(controls_for_or_temp): # Unflip controls (reverse order)
                    qc.x(ctrl)
            
            # Reverse 1a
            for k in range(num_nodes - 1, -1, -1): # Reverse order for uncomputation
                if k != node_idx and (k, node_idx) in edge_map:
                    edge_qubit = edge_q[edge_map[(k, node_idx)]]
                    qc.x(visited_q[k]) # Re-invert visited_q[k]
                    qc.ccx(edge_qubit, visited_q[k], ancilla_incoming_edge_temp[k])
                    qc.x(visited_q[k]) # Re-uninvert visited_q[k]
        
        qc.barrier(label=f"Kahn_Iter_{iteration}_Mark_Visited")

        # Step 2: Mark nodes as visited
        # If ancilla_zero_indegree[node_idx] is 1, it means this node *can* be visited.
        # And it shouldn't already be visited.
        # Use ancilla_or_temp[0] as a temporary for this CCX
        for node_idx in range(num_nodes):
            qc.x(visited_q[node_idx]) # Flip visited_q so unvisited is 1
            # Controls: ancilla_zero_indegree[node_idx] (must be 1) AND visited_q[node_idx] (must be 1 for unvisited)
            qc.ccx(ancilla_zero_indegree[node_idx], visited_q[node_idx], ancilla_or_temp[0])
            qc.cx(ancilla_or_temp[0], visited_q[node_idx]) # Flip visited_q if it should be visited
            # Uncompute the temporary
            qc.ccx(ancilla_zero_indegree[node_idx], visited_q[node_idx], ancilla_or_temp[0])
            qc.x(visited_q[node_idx]) # Flip visited_q back to original sense (0=unvisited)
        
        qc.barrier(label=f"Kahn_Iter_{iteration}_Uncompute_ZeroIndegree")
        # Step 3: Uncompute ancilla_zero_indegree for the next iteration.
        # This reverses the logic of how ancilla_zero_indegree was set in Step 1.
        # This loop needs to reverse the exact operation that set each ancilla_zero_indegree[node_idx]
        
        for node_idx in range(num_nodes - 1, -1, -1): # Uncompute in reverse order
            # Reverse the CCX that set ancilla_zero_indegree[node_idx]
            # This requires `ancilla_or_temp[0]` and `visited_q[node_idx]` to be in their state *before* that CCX.
            # This is the trickiest part for iterative reversible algorithms.
            # Assuming visited_q maintains its cumulative state, we apply the inverse.
            qc.x(visited_q[node_idx]) # Re-invert visited_q[node_idx] for reversal
            qc.ccx(ancilla_or_temp[0], visited_q[node_idx], ancilla_zero_indegree[node_idx])
            qc.x(visited_q[node_idx]) # Un-invert visited_q[node_idx]
            
            # The `ancilla_or_temp[0]` itself was uncomputed in 1d for each node_idx.
            # So, this loop for uncomputing `ancilla_zero_indegree` should effectively
            # undo the `ancilla_zero_indegree` setting without touching other registers.
            # This is still very challenging due to the iterative state updates.
            # For a proper quantum oracle, ideally, `ancilla_zero_indegree` is reset to 0
            # by exactly reversing how it was set for *that specific iteration's calculation*.
            
            # If the calculation of `ancilla_zero_indegree` was self-contained and reset its temp ancillas,
            # then a simple reverse of the final setting gate here is what's needed.
            # Given the deep nesting, it's very hard to guarantee correctness.
            pass # Placeholder for complex cumulative uncomputation of ancilla_zero_indegree
                 # This needs to be exactly the inverse of the computation in step 1c.
                 # The registers `ancilla_or_temp[0]` and `visited_q[node_idx]` need to be
                 # in the state they were in when `ancilla_zero_indegree[node_idx]` was set.
    
    qc.barrier(label="Final_Acyclicity_Check")
    # Final check: Acyclic if all nodes are visited.
    # To check if all are 1:
    qc.x(acyclic_ok_a[0]) # Set target to 1. If any visited_q is 0, it will flip to 0.
    # Use ctrl_state='0' * num_nodes to check if all visited_q are NOT 1.
    # We want acyclic_ok_a[0] to be 1 IF AND ONLY IF all visited_q are 1.
    # Current method: mcx(visited_q[:], acyclic_ok_a[0]) flips if ALL controls are 1.
    # We want: acyclic_ok_a[0] = AND(visited_q[0], visited_q[1], ...)
    # Correct way to set acyclic_ok_a to 1 if all visited_q are 1:
    qc.mcx(visited_q[:], acyclic_ok_a[0]) # If all visited_q are 1, acyclic_ok_a[0] flips.
                                          # Initial state of acyclic_ok_a[0] should be 0.
                                          # So, if all visited_q are 1, it becomes 1. If not, it stays 0.

    # Final Uncomputation of visited_q and other ancillas:
    # This is the most complex part of the entire oracle.
    # You need to reverse the *cumulative* effect on `visited_q` from all iterations.
    # And all `ancilla_zero_indegree`, `ancilla_incoming_edge_temp`, `ancilla_or_temp`
    # must be returned to their initial |0> state.
    # The current `pass` in Step 3 uncomputation is a major missing piece.
    # For `ancilla_incoming_edge_temp` and `ancilla_or_temp`, their uncomputation is done per node_idx
    # within the innermost loop, which is good.
    # But `ancilla_zero_indegree` and especially `visited_q` are cumulative state, making reversal tough.

    # A robust solution for the uncomputation of `visited_q` might involve
    # creating a copy of `visited_q` at the start of each iteration, using it, and then
    # uncomputing back to the original `visited_q` before the next iteration.
    # This creates a significant number of ancillas.

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
    
    cnf_oracle = build_cnf_oracle(cnf, n_variables, n_clauses)
    acyclicity_oracle = build_acyclicity_oracle(num_nodes, node_names, causal_dict, causal_to_cnf_map, n_variables)
    
    edge_q = QuantumRegister(n_variables, name='edge')
    clause_a = QuantumRegister(n_clauses, name='clause')
    cnf_ok_a = QuantumRegister(1, name='cnf_ok')
    visited_q = QuantumRegister(num_nodes, name='visited')
    # These are the registers that match the internal oracle definition now
    ancilla_zero_indegree = QuantumRegister(num_nodes, name='anc_zero_indeg')
    ancilla_path_exists = QuantumRegister(num_nodes + 1, name='anc_path_exists') # Corrected size +1
    acyclic_ok_a = QuantumRegister(1, name='acyclic_ok')
    final_ok_a = QuantumRegister(1, name='final_ok')
    c_reg = ClassicalRegister(n_variables, name='c')

    # --- Create the main QuantumCircuit ---
    qc = QuantumCircuit(edge_q, clause_a, cnf_ok_a, visited_q, 
                        ancilla_zero_indegree, ancilla_path_exists, # Added correctly
                        acyclic_ok_a, final_ok_a, c_reg)
    
    qubit_map = {
        'edge': edge_q, 'final_ok': final_ok_a[0],
        'cnf_ok': cnf_ok_a[0], 'acyclic_ok': acyclic_ok_a[0],
        'cnf_oracle_qubits': edge_q[:] + clause_a[:] + cnf_ok_a[:],
        'acyclic_oracle_qubits': edge_q[:] + visited_q[:] + ancilla_zero_indegree[:] + ancilla_path_exists[:] + acyclic_ok_a[:]
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
    
    qc.measure(edge_q, c_reg)
    
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
