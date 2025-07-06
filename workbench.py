from SAT.quantum import solveQuantumSAT
from SAT.classical import solveClassicalSAT
from SAT.fixedPointQuantum import solveFixedQuantunSAT

import matplotlib.pyplot as plt
import mpmath as mpm

new_cnf = [[1, 5, 2], [-1, -5, 2], [-1, -5, -2], [3, 8, 4], [-3, -8, 4], [-3, -8, -4], [6, 9, 7], [-6, -9, 7], [-6, -9, -7]]
# new_cnf = [[1, 2], [-1, 3], [-2, -3], [1, 3]]
# new_cnf = [[2, 3], [1, 4]]

is_sat, model = solveClassicalSAT(new_cnf)

# is_sat, quantum_solutions = solveQuantumSAT(new_cnf, debug=True)
is_sat, sol = solveFixedQuantunSAT(new_cnf, 8, mpm.sqrt(0.1), debug=True, simulation=True)
# is_sat, sol = solvePiThirdQuantumSAT(new_cnf, debug=True)
# print(model)

# print(quantum_solutions)

# if model in quantum_solutions:
#     print("The standard quantum solution is correct")
# else:
#     print("The standard quantum solution is incorrect")
    
# if model in sol:
#     print("The fixed-point quantum solution is correct")
# else:
#     print("The fixed-point quantum solution is incorrect")

# new_cnf = [[1, 2], [-1, 3], [-2, -3], [1, 3]]
# new_cnf = [[1, 2], [-1, -2]]
# max_rep = 8
# probs = []
# meanOthers = []
# shouldBeOne = []

# # create a dict with the solutions where the key is the solution and the value is the array
# # of the probabilities of each repetition
# solutions = {}

# for i in range(1, max_rep):
#     is_sat, sol = solveFixedQuantunSAT(new_cnf, i, mpm.sqrt(0.1), debug=True)
    
#     # print(sol)
#     print(sol['101'])
    
#     # count the prob of '101'
#     true_sol = '101'
#     if true_sol in sol:
#         probs.append(sol[true_sol])
#     else:
#         probs.append(0)  # Solution not found in this iteration
    
#     # calculate the mean of all the other solutions
#     mean = 0
#     for key, value in sol.items():
#         if key != true_sol:
#             mean += value
    
#     shouldBeOne.append(mean)        
#     mean /= len(sol)
#     meanOthers.append(mean)
    
#     # Keep track of each solution probability
#     for key, value in sol.items():
#         if key not in solutions:
#             solutions[key] = []
#         solutions[key].append(value)
    

    
# # print(probs)
# plt.plot(probs)
# plt.plot(meanOthers)
# plt.legend(["101 probability"])
# plt.savefig("debug/pi-third-quantum.png")
# plt.close()

# # plot in another plot the probabilities of each solution
# for key, value in solutions.items():
#     plt.plot(value, label=key)
# plt.plot(shouldBeOne)
# plt.legend()
# plt.savefig("debug/fixed-point-quantum-solutions.png")
# plt.close()
