from SAT.quantum import solveQuantumSAT
from SAT.classical import solveClassicalSAT
from SAT.fixedPointQuantum import solveFixedQuantunSAT

new_cnf = [[1, 5, 2], [-1, -5, 2], [-1, -5, -2], [3, 8, 4], [-3, -8, 4], [-3, -8, -4], [6, 9, 7], [-6, -9, 7], [-6, -9, -7]]
# new_cnf = [[2, 3], [1, 4]]

# is_sat, model = solveClassicalSAT(new_cnf)

# is_sat, quantum_solutions = solveQuantumSAT(new_cnf, debug=True)

# print(model)

# print(quantum_solutions)

# if model in quantum_solutions:
#     print("The quantum solution is correct")
# else:
#     print("The quantum solution is incorrect")

solveFixedQuantunSAT(new_cnf, debug=True)