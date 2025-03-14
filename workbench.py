from SAT.quantum import solveQuantumSAT
from SAT.classical import solveClassicalSAT
from SAT.fixedPointQuantum import solveFixedQuantunSAT

import matplotlib.pyplot as plt

# new_cnf = [[1, 5, 2], [-1, -5, 2], [-1, -5, -2], [3, 8, 4], [-3, -8, 4], [-3, -8, -4], [6, 9, 7], [-6, -9, 7], [-6, -9, -7]]
# new_cnf = [[2, 3], [1, 4]]

# is_sat, model = solveClassicalSAT(new_cnf)

# is_sat, quantum_solutions = solveQuantumSAT(new_cnf, debug=True)

# print(model)

# print(quantum_solutions)

# if model in quantum_solutions:
#     print("The quantum solution is correct")
# else:
#     print("The quantum solution is incorrect")

# new_cnf = [[1, 2], [-1, 3], [-2, -3], [1, 3]]
new_cnf = [[1, 2], [-1, -2]]
max_rep = 10
probs = []
meanOthers = []
shouldBeOne = []

# create a dict with the solutions where the key is the solution and the value is the array
# of the probabilities of each repetition
solutions = {}

for i in range(5, max_rep):
    sol = solveFixedQuantunSAT(new_cnf, i, debug=True)
    
    # count the prob of '101'
    true_sol = '101'
    if true_sol in sol:
        probs.append(sol[true_sol])
    else:
        probs.append(0)  # Solution not found in this iteration
    
    # calculate the mean of all the other solutions
    mean = 0
    for key, value in sol.items():
        if key != true_sol:
            mean += value
    
    shouldBeOne.append(mean)        
    mean /= len(sol) - 1
    meanOthers.append(mean)
    
    # Keep track of each solution probability
    for key, value in sol.items():
        if key not in solutions:
            solutions[key] = []
        solutions[key].append(value)
    

    
# print(probs)
plt.plot(probs)
plt.plot(meanOthers)
plt.legend(["101", "Mean of the others"])
plt.savefig("debug/fixed-point-quantum.png")
plt.close()

# plot in another plot the probabilities of each solution
for key, value in solutions.items():
    plt.plot(value, label=key)
plt.plot(shouldBeOne)
plt.legend()
plt.savefig("debug/fixed-point-quantum-solutions.png")
plt.close()
