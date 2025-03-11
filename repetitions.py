import numpy as np
import math
import matplotlib.pyplot as plt

ranger = 10

x = np.arange(ranger).tolist()
y = [np.pi/4 * math.sqrt(2**n) for n in range(0, ranger)]

y_rounded = [math.ceil(n) for n in y]

y_classical = [2**n for n in range(0, ranger - 5)]

plt.scatter(x = x, y = y, label="Floating point iterations")
plt.scatter(x = x, y = y_rounded, label="Integer iterations")
plt.scatter(x = x[:-5], y = y_classical, label="Classical iterations")
plt.xlabel("Number of variables")
plt.ylabel("Iterations")
plt.legend()
plt.yticks(y_rounded)
plt.grid(True)
plt.savefig("miscellanous/repetitions.png")

