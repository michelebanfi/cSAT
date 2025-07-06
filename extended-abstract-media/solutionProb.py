import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SAT.fixedPointQuantum import solveFixedQuantunSAT
import matplotlib.pyplot as plt

new_cnf = [[1, 2], [-1, 3], [-2, -3], [1, 3]]

maxrep = 20
probs = []

for i in range(1, 2):
    is_sat, sol = solveFixedQuantunSAT(new_cnf, i, 0.1, debug=True, simulation=True)
    # print(sol)
    probs.append(sol['0000101'])
    
# for i in range(1, maxrep):
#     print(f"({i}, {probs[i-1]}),", end=' ')
    
# plt.plot(list(range(1, maxrep)), probs)
# plt.xlabel("Repetitions")
# plt.ylabel("Solution probability")
# plt.title("Solution probability as a function of repetitions")
# plt.savefig("extended-abstract-media/media/probabilities.png")
# plt.close()

# plt.plot(list(range(1, maxrep)), probs)
# plt.xlabel("Repetitions")
# plt.ylabel("Solution probability")
# plt.ylim(0.9, 1)
# plt.title("Solution probability as a function of repetitions")
# plt.savefig("extended-abstract-media/media/probabilities-cutted.png")
# plt.close()