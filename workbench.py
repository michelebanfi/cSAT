from SAT.quantum import solveQuantumSAT

new_cnf = [[1, 3], [2, 5], [4, 6]]

is_sat, quantum_solutions = solveQuantumSAT(new_cnf, debug=False)

print(quantum_solutions)