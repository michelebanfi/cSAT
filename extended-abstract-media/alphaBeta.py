import numpy as np
import matplotlib.pyplot as plt
import mpmath as mpm

l = 20
delta = np.sqrt(0.1)
L = 2 * l + 1

def chebyshev(L, x):
    return mpm.cos(L * mpm.acos(x))

gamma_inv = chebyshev(1/L, 1/delta)
omega = 1 - chebyshev(1/L, 1/delta)**(-2)
gamma = 1/gamma_inv
# print(f"DEBUG: gamma_inv={gamma_inv}, gamma={gamma}")

alpha_values = mpm.zeros(1, l)
beta_values = mpm.zeros(1, l)
for i in range(l):
    alpha_values[i] = 2*mpm.acot(mpm.tan(2*mpm.pi*(i+1)/L) * mpm.sqrt(1-1/gamma_inv**2))
    beta_values[l-(i+1)+1-1] = -alpha_values[i]
    
gamma_inv = np.array([gamma_inv], dtype=complex)[0].real
omega = np.array([omega], dtype=complex)[0].real
alpha_values = np.array(alpha_values.tolist()[0], dtype=complex).real
beta_values = np.array(beta_values.tolist()[0], dtype=complex).real

plt.scatter(y=alpha_values, x=list(range(l)), label="Alpha")
plt.scatter(y=beta_values,  x=list(range(l)), label="Beta")
plt.title("Alpha and Beta values")
plt.xlabel("Iterations")
plt.ylabel("Values")
plt.legend()
plt.savefig("extended-abstract-media/media/alphas-betas.png")
plt.close()