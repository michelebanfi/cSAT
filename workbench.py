from SAT.quantum import solveQuantumSAT
from SAT.classical import solveClassicalSAT

new_cnf = [[2, 3], [1, 4], [-1, 3]]

is_sat, model = solveClassicalSAT(new_cnf)

is_sat, quantum_solutions = solveQuantumSAT(new_cnf, debug=True)

print(model)

print(quantum_solutions)

if model in quantum_solutions:
    print("The quantum solution is correct")
else:
    print("The quantum solution is incorrect")