import numpy as np
import mpmath as mpm
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
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
    
def oracle(qc, n_variables, beta, cnf, n):
    # print(f"DEBUG: n_variables={n_variables}, cnf={cnf}, n={n}")
    for i, clause in enumerate(cnf):
        # print(f"DEBUG: clause={clause}, i={i}, n_variables+i={n_variables+i}")
        get_repr(qc, False, clause, n_variables + i)
    qc.barrier()

    qc.mcx(list(range(n_variables, n)), n)
    # qc.x(n)
    # qc.mcp(beta, list(range(n_variables,n)), n)
    qc.p(beta, n)
    #qc.x(n)
    qc.mcx(list(range(n_variables, n)), n)

    qc.barrier()
    # Uncompute ancilla qubits
    for i in range(len(cnf)-1, -1, -1):
        get_repr(qc, True, cnf[i], n_variables + i)
    
    qc.barrier()
    
def create_circuit(qc, n_variables, cnf, n, alpha, beta):
    qc.barrier()
    oracle(qc, n_variables, beta,  cnf, n)
    qc.barrier()
    # qc.p(beta, n)
    # qc.barrier()
    # oracle(qc, n_variables, beta, cnf, n)
    # qc.barrier()
    
    qc.h(list(range(n_variables)))
    qc.barrier()
    qc.x(list(range(n_variables - 1)))
    qc.barrier()
    
    qc.p(-alpha/2, n_variables - 1)
    qc.barrier()
    qc.mcx(list(range(n_variables - 1)), n_variables - 1)
    qc.mcx(list(range(n_variables - 1)), n)
    qc.barrier()
    
    qc.p(-alpha/2, n_variables - 1)
    qc.p(-alpha/2, n)
    qc.barrier()
    qc.mcx(list(range(n_variables - 1)), n_variables - 1)
    qc.mcx(list(range(n_variables - 1)), n)
    
    qc.p(alpha, n_variables - 1)
    
    qc.barrier()
    qc.x(list(range(n_variables - 1)))
    qc.barrier()
    qc.h(list(range(n_variables)))
    qc.barrier()
    

def createCircuit(n_variables, l, cnf, n, debug, delta):
    qc = QuantumCircuit(n + 1)
    
    qc.h(list(range(n_variables)))
    
    # L = 2l + 1
    L = 2 * l + 1
    
    gamma_inv = chebyshev(1/L, 1/delta)
    omega = 1 - chebyshev(1/L, 1/delta)**(-2)
    gamma = 1/gamma_inv
    # print(f"DEBUG: gamma_inv={gamma_inv}, gamma={gamma}")
    
    alpha_values = mpm.zeros(1, l)
    beta_values = mpm.zeros(1, l)
    for i in range(l):
        alpha_values[i] = 2*mpm.acot(mpm.tan(2*mpm.pi*(i+1)/L) * mpm.sqrt(1-1/gamma_inv**2))
        beta_values[l-(i+1)+1-1] = -alpha_values[i]
        
    gamma_inv = np.array([gamma_inv], dtype=complex)[0].real
    omega = np.array([omega], dtype=complex)[0].real
    alpha_values = np.array(alpha_values.tolist()[0], dtype=complex).real
    beta_values = np.array(beta_values.tolist()[0], dtype=complex).real

    for i in range(l):
        create_circuit(qc, n_variables, cnf, n, alpha_values[i], beta_values[i])
    
    
    # plot alphas and betas
    if debug: 
        plt.plot(alpha_values, label="Alpha")
        plt.plot(beta_values, label="Beta")
        plt.legend()
        plt.savefig("debug/alphas-betas.png")
        plt.close()
    
    return qc

def run_on_ibm(qc):
    service = QiskitRuntimeService()

    backend = service.least_busy(operational=True, simulator=False)
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=0)
    grover = pm.run(qc)

    # with Session(backend=backend) as session:
    sampler = SamplerV2(mode=backend)
    job = sampler.run([grover], shots=10024)
    pub_result = job.result()
    print(f"Sampler job ID: {job.job_id()}")
    counts = pub_result[0].data.meas.get_counts()
    
    probabilities = {key: value / sum(counts.values()) for key, value in counts.items()}

    return probabilities

def solveFixedQuantumSAT(cnf, l_iterations, delta, debug=False, simulation=True):
    
    # as usual structural check for the CNF
    structural_check(cnf)
    
    variables = set()
    for clause in cnf:
        for var in clause:
            variables.add(abs(var))
            
    n_variables = len(variables)
    n_clauses = len(cnf)
    
    n = n_variables + n_clauses
    
    print(f"LOG: Circuit with {n} qubits")
    
    if n > 20 and simulation:
        print("LOG: Too many qubits, aborting")
        return False, []
    
    qc = createCircuit(n_variables, l_iterations, cnf, n, debug, delta=delta)
    
    print(f"LOG: Circuit created, circuit depth: {qc.depth()}")
    
    qc.measure_all()
    
    if debug:
        circuit_drawer(qc, output="mpl")
        plt.savefig("debug/fixed-circuit.png")
        plt.close()
    
    qc = RemoveBarriers()(qc)
    optimized_qc = transpile(qc, optimization_level=3)
    
    if simulation:
        result = Sampler().run([optimized_qc], shots=1024).result()
        counts = result.quasi_dists[0]
        counts = counts.binary_probabilities(num_bits=n)
    else:
        # Run on IBM quantum hardware
        print("LOG: Running on IBM Quantum Hardware...")
        counts = run_on_ibm(optimized_qc)
    
    # this statememnt is just for printing the solutions as function of the repetitions
    # return True, counts
    
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
            # print(f"DEBUG: bitstring={bitstring}, prob={prob}")
            # remove the first n_clauses bits
            bitstring = bitstring[n_clauses:]
            # print(f"DEBUG: bitstring={bitstring}")
            # reverse the bitstring ordering
            bitstring = bitstring[::-1]
            # print(f"DEBUG: bitstring={bitstring}")
            solution = []
            for i in range(n_variables):
                # print(i)
                var_num = i + 1  # Convert to 1-indexed
                if bitstring[i] == '0':
                    solution.append(-var_num)
                else:
                    solution.append(var_num)
            solutions.append(solution)
    
    # print(f"DEBUG: solutions={solutions}")
        
    return is_sat, solutions