from SAT.quantum import solveQuantumSAT

new_cnf = [[2, 3], [1, 4]]

is_sat, quantum_solutions = solveQuantumSAT(new_cnf)

print(quantum_solutions)