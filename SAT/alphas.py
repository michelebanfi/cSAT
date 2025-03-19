import mpmath as mpm
import numpy as np

from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

def Chbshv_poly(L, x): # Lth Chebyshev polynomial of the first kind
    return mpm.cos(L * mpm.acos(x))

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
    

def solveFP(cnf, reps, d, debug=False):
    variables = set()
    for clause in cnf:
        for var in clause:
            variables.add(abs(var))
            
    n_variables = len(variables)
    n_clauses = len(cnf)
    
    n = n_variables + n_clauses
    
    l = reps
    L = 2*l + 1
    
    gammaInverse = Chbshv_poly(1/L, 1/d)
    omega = 1 - Chbshv_poly(1/L, 1/d)**(-2)
    
    alpha = mpm.zeros(1, l)
    beta = mpm.zeros(1, l)
    
    for i in range(l):
        alpha[i] = 2*mpm.acot(mpm.tan(2*mpm.pi*(i+1)/L) * mpm.sqrt(1-1/gammaInverse**2))
        beta[l-(i+1)+1-1] = -alpha[i]
        
    gammaInverse = np.array([gammaInverse], dtype=complex)[0].real
    omega = np.array([omega], dtype=complex)[0].real
    alpha = np.array(alpha.tolist()[0], dtype=complex).real
    beta = np.array(beta.tolist()[0], dtype=complex).real
    
    print(gammaInverse, omega, alpha, beta)
    
    qc = QuantumCircuit(n + 1)
    
    qc.h(list(range(n_variables)))
    # for i in range(n):
    #     qc.h(n-i)
    
    for i in range(reps):
        oracle(qc, n_variables, cnf, n)
        qc.p(beta[i], n)
        qc.barrier()
        oracle(qc, n_variables, cnf, n)
        qc.h(list(range(n_variables)))
        # for q in range(n):
        #     qc.h(n-q)
        qc.barrier()
        # for q in range(n-1):
        #     qc.x(n-q)
        qc.x(list(range(n_variables)))
        qc.barrier()
        
        qc.p(-alpha[i]/2, 1)
        qc.mcx(list(range(n, 1, -1)), 1)
        qc.mcx(list(range(n, 1, -1)), 0)
        qc.p(-alpha[i]/2, 1)
        qc.p(-alpha[i]/2, 0)
        qc.mcx(list(range(n, 1, -1)), 1)
        qc.mcx(list(range(n, 1, -1)), 0)
        for q in range(n-1):
            qc.x(n-q)
        qc.p(alpha[i], 1)
        qc.barrier()
        qc.h(list(range(n_variables)))
        
    circuit_drawer(qc, output='mpl')
    plt.savefig('debug/circuit.png')
    plt.close()
        
if __name__ == '__main__' :
    solveFP([[1, 2], [-1, 3], [-2, -3], [1, 3]], 1, 0.9)