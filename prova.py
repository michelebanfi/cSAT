from qiskit.quantum_info.operators import Operator, Pauli
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
import numpy as np
import matplotlib.pyplot as plt


theta=np.pi/3
# oracle
Ora = Operator([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j*theta)]])

# diffuser
Rs = Operator([[np.exp(1j*theta), 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc            
            
def fixed_point_grover(N_iter):


    if N_iter==1:

        grover_circuit = QuantumCircuit(n)
        grover_circuit = initialize_s(grover_circuit, [0,1])

        ## 1st iteration    
        # Oracle
        grover_circuit.append(Ora,[0,1])
        grover_circuit.draw()
        
        #(U_s)
        grover_circuit.h([0,1]) 
        grover_circuit.append(Rs,[0,1])
        grover_circuit.h([0,1])
    
        return grover_circuit
    
    else:
         
        grover_circuit = QuantumCircuit(n)
        op=Operator(fixed_point_grover(N_iter-1))
        adop=op.adjoint()
        
        
        grover_circuit.append(op,[0,1])
        
        grover_circuit.append(Ora,[0,1])
        grover_circuit.append(adop,[0,1])

        #(U_s)
        grover_circuit.append(Rs,[0,1])
        grover_circuit.append(op,[0,1])
        
        
        return grover_circuit
    
n=2
fixedgrover=fixed_point_grover(2)

target_vector = np.array([0.0, 0.0, 0.0, 1.0])

fidelity=[]

state_vecs=[]

sol = []

for i in range(1, 5):

    fixedgrover=fixed_point_grover(i)
    fixedgrover.measure_all()
    result = Sampler().run([fixedgrover], shots=1024).result()

    counts=result.quasi_dists[0]
    counts = counts.binary_probabilities(num_bits=n)
    # print(counts)
    
    sol.append(counts['11'])

print(sol)
# fixedgrover.draw()
# fixedgrover.draw(output='mpl')
# plt.savefig("debug/prova.png")