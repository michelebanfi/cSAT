import numpy as np
import math
import matplotlib.pyplot as plt

# x = np.arange(1, 30, step=1)
# np.pi/4 * math.sqrt(2**n_variables)
x = np.arange(10).tolist()
y = [np.pi/4 * math.sqrt(2**n) for n in range(0, 10)]

y_rounded = [math.ceil(n) for n in y]

plt.scatter(x = x, y = y, label="Floating point")
plt.scatter(x = x, y = y_rounded, label="Integers")
plt.legend()
plt.yticks(y_rounded)
plt.grid(True)
plt.show()

