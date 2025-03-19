from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Sampler
from qiskit.quantum_info.operators import Operator
from qiskit.circuit import ClassicalRegister
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

theta = np.pi/4 

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
    
    qc.mcx(control_qubits, i)
    
    if not is_inv:
        qc.x(i)
    
    for var_idx in flipped_qubits:
        qc.x(var_idx)
    
    qc.barrier()

def oracle_circuit(n_variables, cnf):
    n_clauses = len(cnf)
    n_qubits = n_variables + n_clauses + 1
    
    qc = QuantumCircuit(n_qubits)
    
    for i, clause in enumerate(cnf):
        get_repr(qc, False, clause, n_variables + i)

    qc.mcp(theta, list(range(n_variables, n_variables + n_clauses)), n_qubits - 1)
    qc.barrier()

    # for i in reversed(range(n_clauses)):
    for i in range(len(cnf)-1, -1, -1):
        get_repr(qc, True, cnf[i], n_variables + i)
    
    return qc

def diffuser_circuit(n_variables, n_qubits):
 # Adjust based on iteration if needed
    
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_variables))
    qc.x(range(n_variables))
    
    # Use all variable qubits as controls and an ancilla as target (last qubit)
    target = n_qubits - 1  # Using result qubit as target (ensure it's |0>)
    qc.h(target)
    qc.mcp(theta, list(range(n_variables)), target)
    qc.h(target)
    
    qc.x(range(n_variables))
    qc.h(range(n_variables))
    qc.barrier()
    
    return qc

def initialize_s(n_qubits, n_variables):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_variables))
    qc.barrier()
    return qc

def fixed_point_grover_sat(n_variables, cnf, n_iterations):
    n_clauses = len(cnf)
    n_qubits = n_variables + n_clauses + 1
    
    if n_iterations == 1:
        grover_circuit = initialize_s(n_qubits, n_variables)
        oracle_circ = oracle_circuit(n_variables, cnf)
        grover_circuit = grover_circuit.compose(oracle_circ)
        diffuser_circ = diffuser_circuit(n_variables, n_qubits)
        grover_circuit = grover_circuit.compose(diffuser_circ)
        return grover_circuit
    else:
        grover_circuit = QuantumCircuit(n_qubits)
        prev_circuit = fixed_point_grover_sat(n_variables, cnf, n_iterations-1)
        op = Operator(prev_circuit)
        adop = op.adjoint()
        
        grover_circuit.compose(prev_circuit, inplace=True)
        grover_circuit.compose(oracle_circuit(n_variables, cnf), inplace=True)
        grover_circuit.append(adop, range(n_qubits))
        grover_circuit.compose(diffuser_circuit(n_variables, n_qubits), inplace=True)
        grover_circuit.compose(prev_circuit, inplace=True)
        
        return grover_circuit

# Example usage
if __name__ == "__main__":
    cnf = [[1, 2], [-1, 3], [-2, -3], [1, 3]]
    n_variables = 3
    
    # Correct measurement: measure only variable qubits
    for i in range(2, 10):  # Test 1-2 iterations
        fixed_grover = fixed_point_grover_sat(n_variables, cnf, i)
        
        # Measure only the variable qubits (0, 1, 2)
        cr = ClassicalRegister(n_variables)
        fixed_grover.add_register(cr)
        fixed_grover.measure(range(n_variables), cr)
        
        # plot circuit
        fixed_grover.draw(output='mpl')
        plt.savefig(f'debug/prova-circuit-{i}.png')
        plt.close()
        
        # Transpile and run
        result = Sampler().run(fixed_grover, shots=1024).result()
        counts = result.quasi_dists[0].binary_probabilities(num_bits=n_variables)
        plot_histogram(counts)
        plt.savefig(f'debug/prova-histogram-{i}.png')
        plt.close()
        
        # Check for solution '101' (x1=1, x2=0, x3=1)
        solution_prob = counts.get('101', 0)
        print(f"Iteration {i}: Solution probability = {solution_prob}")