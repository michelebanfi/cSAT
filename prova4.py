from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PhaseOracle
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit_algorithms import AmplificationProblem, Grover
import numpy as np
from sympy import symbols, sympify, to_dnf
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler

class FixedPointGroverSAT:
    """
    Implementation of the Fixed-Point pi/3 Grover's algorithm for boolean satisfiability problems.
    This version avoids the overshooting problem of traditional Grover's algorithm.
    """
    
    def __init__(self, formula_str, variable_names=None):
        """
        Initialize with a boolean formula in string format.
        
        Args:
            formula_str (str): Boolean formula in string format
            variable_names (list): Optional list of variable names. If None, variables are extracted from formula
        """
        self.formula_str = formula_str
        
        # Extract variables if not provided
        if variable_names is None:
            # Parse formula to extract variables
            self.variables = self._extract_variables(formula_str)
        else:
            self.variables = variable_names
            
        self.n_qubits = len(self.variables)
        self.oracle = None
        self.state_preparation = None
        self.diffuser = None
        
    def _extract_variables(self, formula_str):
        """Extract variable names from the formula string"""
        # Use sympy to parse the formula and extract symbols
        sym_formula = sympify(formula_str, evaluate=False)
        sym_vars = list(sym_formula.free_symbols)
        return sorted([str(v) for v in sym_vars])
    
    def _create_oracle(self):
        """Create oracle circuit from the boolean formula"""
        # Convert formula to DNF for Qiskit's PhaseOracle
        sym_vars = symbols(" ".join(self.variables))
        sym_formula = sympify(self.formula_str, evaluate=False)
        
        # Convert formula to DNF format required by PhaseOracle
        dnf_formula = str(to_dnf(sym_formula))
        
        # Create PhaseOracle - this applies a -1 phase to states that satisfy the formula
        self.oracle = PhaseOracle(dnf_formula)#expression_mapping=self.variables
        return self.oracle
    
    def _create_state_preparation(self):
        """Create a circuit for uniform superposition state preparation"""
        state_prep = QuantumCircuit(self.n_qubits, name="init")
        for qubit in range(self.n_qubits):
            state_prep.h(qubit)
        self.state_preparation = state_prep
        return state_prep
    
    def _create_diffuser(self):
        """Create the diffuser (reflection about the mean) circuit"""
        diffuser = QuantumCircuit(self.n_qubits, name="diffuser")
        
        # Apply H gates to all qubits
        for qubit in range(self.n_qubits):
            diffuser.h(qubit)
        
        # Apply X gates to all qubits
        for qubit in range(self.n_qubits):
            diffuser.x(qubit)
        
        # Apply multi-controlled Z (reflection about |11...1>)
        if self.n_qubits > 2:
            diffuser.h(self.n_qubits - 1)
            # Use MCX for multi-controlled NOT
            control_qubits = list(range(self.n_qubits - 1))
            target = self.n_qubits - 1
            diffuser.mcx(control_qubits, target)
            diffuser.h(self.n_qubits - 1)
        elif self.n_qubits == 2:
            diffuser.cz(0, 1)
        else:
            diffuser.z(0)
        
        # Apply X gates again
        for qubit in range(self.n_qubits):
            diffuser.x(qubit)
        
        # Apply H gates again
        for qubit in range(self.n_qubits):
            diffuser.h(qubit)
            
        self.diffuser = diffuser
        return diffuser
    
    def _pi_3_fixed_point_iteration(self, circuit):
        """
        Apply a Fixed-Point pi/3 Grover iteration.
        
        This uses a modified approach where we apply the oracle, followed by a diffuser with
        a specific phase shift to avoid overshooting.
        """
        # Create subcircuit for the fixed-point pi/3 iteration
        pi3_circuit = QuantumCircuit(self.n_qubits, name="pi3_iteration")
        
        # Apply oracle 
        pi3_circuit.compose(self.oracle, inplace=True)
        
        # Apply modified diffuser with phase shift
        # Fixed-point approach: use custom phase for diffuser instead of standard -1 phase
        phase_shift = np.exp(1j * np.pi/3)
        
        # Apply the phase rotations before diffuser
        for qubit in range(self.n_qubits):
            pi3_circuit.h(qubit)
            pi3_circuit.x(qubit)
        
        # Apply controlled phase rotation
        if self.n_qubits > 1:
            # Multi-qubit case: use multi-controlled phase
            control_qubits = list(range(self.n_qubits - 1))
            target = self.n_qubits - 1
            
            # Apply Hadamard to target
            pi3_circuit.h(target)
            
            # Apply controlled-Z with modified phase
            pi3_circuit.p(np.pi/3, target)
            pi3_circuit.mcx(control_qubits, target)
            pi3_circuit.p(-np.pi/3, target)
            pi3_circuit.mcx(control_qubits, target)
            
            # Apply Hadamard to target
            pi3_circuit.h(target)
        else:
            # Single qubit case: just apply phase
            pi3_circuit.p(np.pi/3, 0)
        
        # Complete the diffuser
        for qubit in range(self.n_qubits):
            pi3_circuit.x(qubit)
            pi3_circuit.h(qubit)
            
        # Append to main circuit
        circuit.compose(pi3_circuit, inplace=True)
        
        return circuit
    
    def build_circuit(self, iterations=None):
        """
        Build the complete fixed-point Grover circuit
        
        Args:
            iterations (int): Number of iterations to apply. If None, uses optimal iterations.
        
        Returns:
            QuantumCircuit: Complete circuit
        """
        # Create oracle if not already created
        if self.oracle is None:
            self._create_oracle()
        
        # Create state preparation if not already created
        if self.state_preparation is None:
            self._create_state_preparation()
            
        # Create diffuser if not already created
        if self.diffuser is None:
            self._create_diffuser()
        
        # Calculate optimal number of iterations if not specified
        if iterations is None:
            # For Fixed-Point algorithm, we can use more iterations without risk of overshooting
            # A reasonable default is π/4 * sqrt(N) where N is the size of the search space
            iterations = int(np.ceil(np.pi/4 * np.sqrt(2**self.n_qubits)))
            # For small problem sizes, ensure we do at least 2 iterations
            iterations = max(iterations, 2)
        
        # Start with state preparation
        circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        circuit.compose(self.state_preparation, inplace=True)
        
        # Apply fixed-point iterations
        for i in range(iterations):
            self._pi_3_fixed_point_iteration(circuit)
        
        # Add measurements
        circuit.measure(range(self.n_qubits), range(self.n_qubits))
        
        return circuit
    
    def solve(self, iterations=None, shots=1024):
        """
        Run the Fixed-Point pi/3 Grover algorithm to find solutions
        
        Args:
            iterations (int): Number of iterations to apply. If None, uses optimal.
            shots (int): Number of measurements to take
            
        Returns:
            dict: Results dictionary with assignments and their counts
        """
        # Build circuit
        circuit = self.build_circuit(iterations)
        
        # Run simulation
        # simulator = AerSimulator()
        # transpiled_circuit = transpile(circuit, simulator)
        # result = simulator.run(transpiled_circuit, shots=shots).result()
        # counts = result.get_counts()

        optimized_circuit = transpile(circuit, optimization_level=3)
        result = Sampler().run([optimized_circuit], shots=shots).result()
        counts = result.quasi_dists[0]
        counts = counts.binary_probabilities(num_bits=self.n_qubits)
        
        return counts
        
        # # Convert binary outcomes to variable assignments
        # solutions = {}
        # for bitstring, count in counts.items():
        #     # Convert from little-endian to conventional order
        #     bit_list = list(reversed(bitstring))
            
        #     # Create variable assignment dictionary
        #     assignment = {}
        #     for i, var in enumerate(self.variables):
        #         assignment[var] = (bit_list[i] == '1')
                
        #     # Create a string representation of the assignment
        #     assignment_str = ', '.join([f"{var}={val}" for var, val in assignment.items()])
        #     solutions[assignment_str] = count
        
        # return solutions
    
    def verify_solutions(self, results):
        """
        Verify solutions against the original boolean formula
        
        Args:
            results (dict): Results from solve()
            
        Returns:
            dict: Verified results with boolean correctness
        """
        from sympy.logic.boolalg import to_dnf
        from sympy import sympify, symbols
        
        # Parse the formula
        sym_vars = symbols(" ".join(self.variables))
        sym_formula = sympify(self.formula_str, evaluate=False)
        
        # Convert string assignments back to dictionaries
        verified_results = {}
        for assignment_str, count in results.items():
            # Parse the assignment string back to a dictionary
            assignment_dict = {}
            for var_assignment in assignment_str.split(', '):
                var, val = var_assignment.split('=')
                assignment_dict[var] = (val == 'True')
            
            # Substitute values into the formula
            substitution = {symbols(var): val for var, val in assignment_dict.items()}
            is_solution = bool(sym_formula.subs(substitution))
            
            # Add to verified results
            verified_results[assignment_str] = {
                'count': count, 
                'is_solution': is_solution
            }
        
        return verified_results
    
    def visualize_results(self, results, title="Fixed-Point Grover Results"):
        """
        Visualize the results
        
        Args:
            results (dict): Results from solve()
            title (str): Title for the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Sort by counts
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(sorted_results.keys(), sorted_results.values())
        
        # Add percentages on top of bars
        total = sum(sorted_results.values())
        for bar in bars:
            height = bar.get_height()
            percentage = 100 * height / total
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f"{percentage:.1f}%", ha='center', va='bottom', rotation=0)
        
        plt.title(title)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # # Example boolean formula: (a OR b) AND NOT(a AND b) - this is XOR
    # formula = "(a | b) & ~(a & b)"
    
    # # Create and run solver
    # solver = FixedPointGroverSAT(formula)
    # results = solver.solve(shots=1024)
    
    # print("Boolean Formula:", formula)
    # print("\nResults:")
    # for assignment, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{assignment}: {count}")
    
    # # Verify solutions
    # verified_results = solver.verify_solutions(results)
    # print("\nVerified Results:")
    # for assignment, data in verified_results.items():
    #     print(f"{assignment}: Count={data['count']}, Is Solution={data['is_solution']}")
    
    # # Visualize results
    # solver.visualize_results(results)
    # plt.show()
    
    # # Try a more complex formula
    # print("\n\nTesting a more complex formula:")
    # complex_formula = "(a & b & c) | (a & ~b & ~c) | (~a & b & ~c)"
    # complex_solver = FixedPointGroverSAT(complex_formula)
    # complex_results = complex_solver.solve(shots=2048)
    
    # print("Boolean Formula:", complex_formula)
    # print("\nResults:")
    # for assignment, count in sorted(complex_results.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{assignment}: {count}")
    
    # # Verify complex solutions
    # verified_complex = complex_solver.verify_solutions(complex_results)
    # print("\nVerified Results:")
    # for assignment, data in verified_complex.items():
    #     if data['is_solution']:
    #         print(f"{assignment}: Count={data['count']}, Is Solution={data['is_solution']}")
    
    # # Visualize complex results
    # complex_solver.visualize_results(complex_results, "Complex Formula Results")
    # plt.show()
    
    # [[1, 2], [-1, 3], [-2, -3], [1, 3]]
    formula = "(a | b) & (~a | c) & (~b | ~c) & (a | c)"
    solver = FixedPointGroverSAT(formula, variable_names=['a', 'b', 'c'])
    probs = [] 
    for i in range(1, 15):
        results = solver.solve(iterations=i, shots=1024)
        probs.append(results['101'])
        # verified_results = solver.verify_solutions(results)
        # print(f"\n\nResults for {i} iterations:")
        # for assignment, data in verified_results.items():
        #     if data['is_solution']:
        #         print(f"{assignment}: Count={data['count']}, Is Solution={data['is_solution']}")
        
    print(probs)