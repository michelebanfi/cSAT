import mpmath as mpm
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_aer import AerSimulator
from qiskit import ClassicalRegister



def Chbshv_poly(L, x): # Lth Chebyshev polynomial of the first kind
    return mpm.cos(L * mpm.acos(x))

def oracle(qc, n, ancilla, beta):
    qc.x(0)
    qc.x(1)
    qc.mcx([0, 1], n)
    qc.x(n)
    qc.x(0)
    qc.x(1)
    # qc.cx(n, n + ancilla)
    # qc.mcx(list(range(0, n)), n + ancilla)
    #qc.x(n)
    qc.barrier()
    qc.x(2)
    qc.mcx([2, 0], n + 1)
    qc.x(n+1)
    qc.x(2)
    qc.barrier()
    
    qc.mcx([1, 2], n + 2)
    qc.x(n+2)
    qc.barrier()
    
    qc.x(0)
    qc.x(2)
    qc.mcx([0, 2], n + 3)
    qc.x(n+3)
    qc.x(0)
    qc.x(2)
    qc.barrier()
    
    qc.barrier()
    qc.mcx(list(range(n, n + ancilla)), n + ancilla)
    # qc.x(n+ancilla)
    qc.barrier()
    qc.p(beta, n + ancilla)
    qc.barrier()
    # qc.x(n+ancilla)
    qc.mcx(list(range(n, n + ancilla)), n + ancilla)
    qc.barrier()
    
    qc.x(n+3)
    qc.x(0)
    qc.x(2)
    qc.mcx([0, 2], n + 3)
    qc.x(0)
    qc.x(2)
    qc.barrier()
    
    qc.x(n+2)
    qc.mcx([1, 2], n + 2)
    qc.barrier()
    
    qc.x(2)
    qc.x(n+1)
    qc.mcx([2, 0], n + 1)
    qc.x(2)
    qc.barrier()
    
    qc.x(1)
    qc.x(0)
    qc.x(n)
    qc.mcx([0, 1], n)
    qc.x(1)
    qc.x(0)
    
def FP_Grover_circuit(n, itr, d, return_params = True):
    
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
    ancilla = 4
    qc = QuantumCircuit(n + ancilla + 1)
    # Initialize |s>
    for i in range(n):
        qc.h(i) # Measurement order is the reversed qubit order
    for i in range(itr):
        # St(beta)
        qc.barrier()
        oracle(qc, n, ancilla, beta[i]) 
        qc.barrier()
        # qc.p(beta[i], n + ancilla) 
        # qc.barrier()
        # oracle(qc, n, ancilla) 
        # St(alpha)
        qc.barrier()
        
        for q in range(n):
            qc.h(q)
        
        qc.barrier()
        
        for q in range(n-1):
            qc.x(q)
        
        qc.p(-alpha[i]/2, n-1)
        
        qc.mcx(list(range(0, n-1)), n-1)
        qc.mcx(list(range(0, n-1,)), n + ancilla)
        qc.p(-alpha[i]/2, n + ancilla)
        qc.p(-alpha[i]/2, n-1)
        qc.mcx(list(range(0, n-1)), n-1)
        qc.mcx(list(range(0, n-1,)), n + ancilla)
        
        for q in range(n-1):
            qc.x(q)
        
        qc.p(alpha[i], n-1)
        
        qc.barrier()
        for q in range(n):
            qc.h(q)
    if return_params:
        return qc, (gamma_inverse, 1/2**n, omega, alpha, beta)
    else:
        return qc
    
if __name__ == '__main__':
    n = 3
    itr = 1
    d = mpm.sqrt(0.1) 
    for i in range(1, 2):
        qc, (gamma_inverse, lam, omega, alpha, beta) = FP_Grover_circuit(n, i, d)
        # cr = ClassicalRegister(n)
        # qc.add_register(cr)
        # qc.measure(range(n), range(n))
        qc.measure_all()
        circuit_drawer(qc, output='mpl')
        plt.savefig('debug/fixed-reversed-circuit.png')
        plt.close()
        
        qc = RemoveBarriers()(qc)
        optimized_qc = transpile(qc, optimization_level=3)
        
        simulator = AerSimulator()
        result = simulator.run(optimized_qc, shots=1024).result()
        counts = result.get_counts()
        
        # for key, value in counts.items():
        #     # print jus the last n qubits
        #     print(key[-n:], value)
        print(counts)
        print("-----------------")
        
        # print(counts)
        # print(counts['0010'])
        # print(counts['0100']/1024)
        # print(lam, omega)