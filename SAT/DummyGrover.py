import mpmath as mpm
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_aer import AerSimulator



def Chbshv_poly(L, x): # Lth Chebyshev polynomial of the first kind
    return mpm.cos(L * mpm.acos(x))

def oracle(qc,n, indices_to_mark):
    # create a quantum circuit on n qubits
    index_bit =  format(indices_to_mark, "0{:d}b".format(n))
    # print(index_bit)

    for i in range(n):
        if index_bit[i] == '0':
            qc.x(n-i) # Measurement order is the reversed qubit order
    qc.mcx(list(range(n,0,-1)), 0)
    for i in range(n): # Redo the NOT gates applied on control qubits
        if index_bit[i] == '0':
            qc.x(n-i) # Measurement order is the reversed qubit order

    
    
def FP_Grover_circuit(n, indices_to_mark, itr, d, return_params = True):
    # Does not include measurements to allow state tomography
    l = itr
    L = 2*l+1

    gamma_inverse = Chbshv_poly(1/L, 1/d)
    omega = 1 - Chbshv_poly(1/L, 1/d)**(-2)

    alpha =  mpm.zeros(1,l)
    beta = mpm.zeros(1,l)
    for i in range(l): # use i instead of j since python use 1j for sqrt(-1)
        alpha[i] = 2*mpm.acot(mpm.tan(2*mpm.pi*(i+1)/L) * mpm.sqrt(1-1/gamma_inverse**2))
        beta[l-(i+1)+1-1] = -alpha[i]
        
    # print(gamma_inverse, omega, alpha, beta)

    # Convert to numpy
    gamma_inverse = np.array([gamma_inverse], dtype=complex)[0].real
    omega = np.array([omega], dtype=complex)[0].real
    alpha = np.array(alpha.tolist()[0], dtype=complex).real
    beta = np.array(beta.tolist()[0], dtype=complex).real
    
    r = QuantumRegister(n+1)
    qc = QuantumCircuit(r)
    # Initialize |s>
    for i in range(n):
        qc.h(n-i) # Measurement order is the reversed qubit order
    for i in range(itr):
        # St(beta)
        qc.barrier()
        oracle(qc,n, indices_to_mark) # turn state into |T>|1> + sum_i (|w_i>|0>) where w_i are NOT target state, T is the target state
        qc.barrier()
        qc.p(beta[i],0) # when beta[i] = pi, this is simply a Z gate, so only has phase kickback on |T>|1> but not |w_i>|0>
        qc.barrier()
        oracle(qc,n, indices_to_mark)  # to uncompute the ancillary
        # St(alpha)
        qc.barrier()
        
        for q in range(n):
            qc.h(n-q)
        
        qc.barrier()
        
        for q in range(n - 1):
            qc.x(n-q)
        
        qc.barrier()
        
        qc.p(-alpha[i]/2, 1)
        qc.mcx(list(range(n, 1, -1)), 1)
        qc.mcx(list(range(n, 1, -1)), 0)
        qc.p(-alpha[i]/2, 1)
        qc.p(-alpha[i]/2, 0)
        qc.mcx(list(range(n, 1, -1)), 1)
        qc.mcx(list(range(n, 1, -1)), 0)
        for q in range(n - 1):
            qc.x(n-q)
        qc.p(alpha[i], 1)
        qc.barrier()
        for q in range(n):
            qc.h(n-q)
    if return_params:
        return qc, (gamma_inverse, 1/2**n, omega, alpha, beta)
    else:
        return qc
    
if __name__ == '__main__':
    n = 3
    indices_to_mark = 2
    itr = 1
    d = mpm.sqrt(0.1) 
    for i in range(1, 2):
        qc, (gamma_inverse, lam, omega, alpha, beta) = FP_Grover_circuit(n, indices_to_mark, i, d)
        qc.measure_all()
        circuit_drawer(qc, output='mpl')
        plt.savefig('debug/circuit.png')
        plt.close()
        
        qc = RemoveBarriers()(qc)
        optimized_qc = transpile(qc, optimization_level=3)
        
        simulator = AerSimulator()
        result = simulator.run(optimized_qc, shots=1024).result()
        counts = result.get_counts()
        
        # print(counts)
        print(counts['0100']/1024)
        # print(lam, omega)