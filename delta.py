import numpy as np
import matplotlib.pyplot as plt

delta = np.linspace(0.01, 0.95, 30)
# for each delta, calculate 1- 1/(gamma^2)
gamma = [(1 - d**2) for d in delta]
plt.plot(delta, gamma)
plt.xlabel("Delta")
plt.ylabel("1 - 1/(gamma^2)")
plt.grid(True)
plt.savefig("miscellanous/delta.png")