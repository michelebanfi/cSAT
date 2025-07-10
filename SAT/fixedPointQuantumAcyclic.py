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
    Simplified version that's easier to debug.
    Checks if the graph is acyclic by verifying that we can find a valid topological ordering.
    """
    edge_q = QuantumRegister(n_variables, name='edge')
    work_a = AncillaRegister(num_nodes * num_nodes, name='work')  # Working space
    acyclic_ok_a = AncillaRegister(1, name='acyclic_ok')
    
    qc = QuantumCircuit(edge_q, work_a, acyclic_ok_a, name="Acyclicity_Oracle_Simple")
    
    # Build edge mapping
    edge_map = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                key = (node_names[i], node_names[j], 'direct')
                if key in causal_dict and causal_dict[key] in causal_to_cnf_map:
                    cnf_var = causal_to_cnf_map[causal_dict[key]]
                    edge_map[(i, j)] = cnf_var - 1
    
    # For small graphs, we can check all possible cycles directly
    # Check for cycles of length 2 (bidirectional edges)
    work_idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) in edge_map and (j, i) in edge_map:
                # Check if both edges exist
                qc.ccx(edge_q[edge_map[(i, j)]], edge_q[edge_map[(j, i)]], work_a[work_idx])
                work_idx += 1
    
    # Check for cycles of length 3
    for i in range(num_nodes):
        for j in range(num_nodes):
            for k in range(num_nodes):
                if i != j and j != k and k != i:
                    edges = [(i, j), (j, k), (k, i)]
                    if all(edge in edge_map for edge in edges):
                        edge_qubits = [edge_q[edge_map[edge]] for edge in edges]
                        qc.mcx(edge_qubits, work_a[work_idx])
                        work_idx += 1
    
    # The graph is acyclic if NO cycles are found
    if work_idx > 0:
        qc.x(work_a[:work_idx])  # Flip all cycle indicators
        qc.mcx(work_a[:work_idx], acyclic_ok_a[0])  # All must be 1 (no cycles)
        qc.x(work_a[:work_idx])  # Flip back
    else:
        # No possible cycles, always acyclic
        qc.x(acyclic_ok_a[0])
    
    # Uncompute work register
    work_idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) in edge_map and (j, i) in edge_map:
                qc.ccx(edge_q[edge_map[(i, j)]], edge_q[edge_map[(j, i)]], work_a[work_idx])
                work_idx += 1
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            for k in range(num_nodes):
                if i != j and j != k and k != i:
                    edges = [(i, j), (j, k), (k, i)]
                    if all(edge in edge_map for edge in edges):
                        edge_qubits = [edge_q[edge_map[edge]] for edge in edges]
                        qc.mcx(edge_qubits, work_a[work_idx])
                        work_idx += 1
    
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
    incoming_a = QuantumRegister(num_nodes, name='incoming')
    zero_deg_a = QuantumRegister(num_nodes, name='zero_deg')
    acyclic_ok_a = QuantumRegister(1, name='acyclic_ok')
    final_ok_a = QuantumRegister(1, name='final_ok')
    c_reg = ClassicalRegister(n_variables, name='c')
    
    qc = QuantumCircuit(edge_q, clause_a, cnf_ok_a, visited_q, incoming_a, zero_deg_a, acyclic_ok_a, final_ok_a, c_reg)
    
    qubit_map = {
        'edge': edge_q, 'final_ok': final_ok_a[0],
        'cnf_ok': cnf_ok_a[0], 'acyclic_ok': acyclic_ok_a[0],
        'cnf_oracle_qubits': edge_q[:] + clause_a[:] + cnf_ok_a[:],
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
