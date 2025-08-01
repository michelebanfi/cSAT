#%%
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import CPhaseGate
from qiskit_aer import StatevectorSimulator

# --- Configuration ---
NUM_NODES = 3
NUM_EDGES = NUM_NODES * (NUM_NODES - 1)
EDGE_MAP = {(0,1):0, (0,2):1, (1,0):2, (1,2):3, (2,0):4, (2,1):5}

# --- CCRZ Gate Implementation ---
def ccrz_gate(qc, angle, control1, control2, target):
    """Improved doubly-controlled Z rotation with proper decomposition"""
    # Use CP gate (equivalent to deprecated CU1)
    qc.append(CPhaseGate(angle/2), [control2, target])
    qc.cx(control1, control2)
    qc.append(CPhaseGate(-angle/2), [control2, target])
    qc.cx(control1, control2)
    qc.append(CPhaseGate(angle/2), [control1, target])

# --- Oracle Function (Corrected) ---
def build_acyclicity_oracle(edge_qubits, phi_qubits, removed_qubits, global_ancilla, flip_ancilla):
    """
    Constructs the acyclicity oracle with:
    - Proper phase encoding of in-degrees
    - Correct node removal marking
    - Accurate in-degree updates
    - Final DAG verification
    """
    oracle = QuantumCircuit(
        edge_qubits, 
        phi_qubits, 
        removed_qubits, 
        global_ancilla, 
        flip_ancilla,
        name="Acyclicity Oracle"
    )
    
    # 1. Initialize qubits
    oracle.h(phi_qubits)  # Prepare |+> states for phase encoding
    angle = 2 * np.pi / NUM_NODES
    
    # 2. Encode initial in-degrees with phase rotations
    # Node 0 (A): Incoming from B->A (edge2) and C->A (edge4)
    oracle.crz(angle, edge_qubits[EDGE_MAP[(1,0)]], phi_qubits[0])
    oracle.crz(angle, edge_qubits[EDGE_MAP[(2,0)]], phi_qubits[0])
    
    # Node 1 (B): Incoming from A->B (edge0) and C->B (edge5)
    oracle.crz(angle, edge_qubits[EDGE_MAP[(0,1)]], phi_qubits[1])
    oracle.crz(angle, edge_qubits[EDGE_MAP[(2,1)]], phi_qubits[1])
    
    # Node 2 (C): Incoming from A->C (edge1) and B->C (edge3)
    oracle.crz(angle, edge_qubits[EDGE_MAP[(0,2)]], phi_qubits[2])
    oracle.crz(angle, edge_qubits[EDGE_MAP[(1,2)]], phi_qubits[2])
    
    # 3. Iterative node removal
    for _ in range(NUM_NODES):
        # Convert phase to computational basis for testing
        oracle.h(phi_qubits)
        
        # Identify and mark removable nodes
        for j in range(NUM_NODES):
            # Check if node j has zero in-degree (phi_qubits[j] = |0>) 
            # and hasn't been removed yet (removed_qubits[j] = |0>)
            oracle.x(phi_qubits[j])
            oracle.x(removed_qubits[j])
            oracle.ccx(phi_qubits[j], removed_qubits[j], flip_ancilla[0])
            oracle.cx(flip_ancilla[0], removed_qubits[j])
            oracle.ccx(phi_qubits[j], removed_qubits[j], flip_ancilla[0])  # Uncompute
            oracle.x(removed_qubits[j])
            oracle.x(phi_qubits[j])
        
        # Convert back to phase encoding
        oracle.h(phi_qubits)
        
        # Update in-degrees of successors
        for j in range(NUM_NODES):
            for l in range(NUM_NODES):
                if j == l: 
                    continue
                edge_idx = EDGE_MAP.get((j, l))
                if edge_idx is not None:
                    # Apply reverse rotation for edges from removed nodes
                    ccrz_gate(
                        oracle, 
                        -angle, 
                        edge_qubits[edge_idx], 
                        removed_qubits[j], 
                        phi_qubits[l]
                    )
    
    # 4. Final verification - check all nodes removed
    oracle.mcx(removed_qubits, global_ancilla[0])
    
    return oracle

# --- Testing Function ---
def test_acyclicity_oracle(graph_state_str):
    """Tests oracle with various graph configurations"""
    # Create quantum registers
    edge_q = QuantumRegister(NUM_EDGES, 'edge')
    phi_q = QuantumRegister(NUM_NODES, 'phi')
    removed_q = QuantumRegister(NUM_NODES, 'removed')
    global_a = QuantumRegister(1, 'global')
    flip_a = QuantumRegister(1, 'flip')
    
    # Build circuit
    test_circuit = QuantumCircuit(edge_q, phi_q, removed_q, global_a, flip_a)
    
    # Initialize edge qubits
    for i, bit in enumerate(graph_state_str):
        if bit == '1':
            test_circuit.x(edge_q[i])
    
    # Add oracle
    oracle = build_acyclicity_oracle(edge_q, phi_q, removed_q, global_a, flip_a)
    test_circuit.compose(oracle, inplace=True)
    
    # Measure global ancilla
    test_circuit.measure_all()
    
    # Simulate
    simulator = StatevectorSimulator()
    result = simulator.run(test_circuit).result()
    statevector = result.get_statevector()
    
    # Calculate probability of DAG detection
    prob_dag = 0
    # Global ancilla is the last qubit (index -1)
    global_idx = len(test_circuit.qubits) - 1
    for i, amp in enumerate(statevector):
        if (i >> global_idx) & 1:  # Check if global ancilla is |1>
            prob_dag += abs(amp) ** 2
    
    # Print results
    print(f"\nGraph: {graph_state_str}")
    print(f"P(DAG) = {prob_dag:.4f}")
    print("Expected:", "DAG" if prob_dag > 0.9 else "Cyclic")

# --- Test Cases ---
if __name__ == "__main__":
    print("Testing Acyclicity Oracle:")
    
    # Valid DAGs
    test_acyclicity_oracle('100100')  # A->B, B->C
    test_acyclicity_oracle('100001')  # A->B, C->B
    test_acyclicity_oracle('010000')  # B->C
    test_acyclicity_oracle('000000')  # No edges
    
    # Cyclic graphs
    test_acyclicity_oracle('100110')  # A->B, B->C, C->B (cycle)
    test_acyclicity_oracle('111111')  # Fully connected


