import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.primitives import Sampler
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.visualization import circuit_drawer
import qiskit

import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

def get_repr(qc, is_inv, clause, i):
    # Keep track of which qubits we flipped due to negation
    flipped_qubits = []
    
    # For each variable in the clause, apply X gates to negated variables
    for var in clause:
        var_idx = abs(var) - 1  # Convert to 0-indexed
        
        # If the variable is negated, flip it
        if var < 0:
            qc.x(var_idx)
            flipped_qubits.append(var_idx)
    
    print(f"DEBUG: flipped")
    
    # Apply X gates to all variables for OR logic implementation
    control_qubits = [abs(var) - 1 for var in clause] # still 0-indexed
    
    print(f"DEBUG: control_qubits={control_qubits}")
    for var_idx in control_qubits:
        qc.x(var_idx)
    
    # Conditional inversion of ancilla qubit
    if is_inv:
        qc.x(i)
    
    # Use multi-controlled-X to implement the clause logic
    qc.mcx(control_qubits, i)
    
    # Conditional inversion of ancilla qubit
    if not is_inv:
        qc.x(i)
    
    # Undo the X gates on all variables
    for var_idx in control_qubits:
        qc.x(var_idx)
    
    # Undo the flips for negated variables
    for var_idx in flipped_qubits:
        qc.x(var_idx)
    
    qc.barrier()

def oracle(qc, n_variables, cnf, n):
    print(f"DEBUG: n_variables={n_variables}, cnf={cnf}, n={n}")
    for i, clause in enumerate(cnf):
        print(f"DEBUG: clause={clause}, i={i}, n_variables+i={n_variables+i}")
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

def solveQuantumSAT(cnf):
    
    # print qiskit version
    print(f"LOG: qiskit version: {qiskit.__version__}")
    
    print(f"LOG: creating circuit")
    
    variables = set()
    for clause in cnf:
        for var in clause:
            variables.add(abs(var))
            
    n_variables = len(variables)
    n_clauses = len(cnf)
    
    n = n_variables + n_clauses
    reps = round(np.pi/4 * math.sqrt(2**n_variables))
    
    qc = QuantumCircuit(n)
    
    qc.h(list(range(n_variables)))
    
    for i in range(reps):
        create_circuit(qc, n_variables, cnf, n)
        
    # Measure variable qubits
    qc.measure_all()
    
    # uncomment just for debugging the circuit (yes, i mean the plotting)
        
    # circuit_drawer(qc, output='mpl')
    # plt.show()
    
    optimized_qc = transpile(qc, optimization_level=3)
    result = Sampler().run([optimized_qc], shots=1024).result()
    # print(f"DEBUG: result={result}")
    
    counts = result.quasi_dists[0]
    
    print(f"DEBUG: counts={counts}")
    
    counts = counts.binary_probabilities(num_bits=n)
    print(f"DEBUG: counts={counts}")
    
    # plot_histogram(counts)
    plt.show()
    solutions = []
    
    if len(counts) == 0:
        return False, []
    else:
        is_sat = True
        for bitstring, prob in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            #print(f"DEBUG: bitstring={bitstring}, prob={prob}")
            if prob > 0.01:  # Only consider significant probabilities
                solution = []
                for i in range(n_variables):
                    var_num = i + 1  # Convert to 1-indexed
                    if bitstring[i] == '0':
                        solution.append(-var_num)
                    else:
                        solution.append(var_num)
                solutions.append(solution)
    
    print(f"DEBUG: solutions={solutions}")
    
        
    return is_sat, solutions