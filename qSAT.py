

# import libraries
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import numpy as np
from colorsys import hls_to_rgb
import re
from sympy.logic.boolalg import is_cnf, to_cnf
import math
from qiskit import transpile
from qiskit.transpiler.passes import RemoveBarriers


# function to plot phases of the states
def plot_statevector_circles(statevector, figsize=(8, 8)):
    num_qubits = int(np.log2(len(statevector)))
    num_states = len(statevector)

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_states)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    # Calculate radius for circles (adjust based on grid size)
    radius = 0.4 / grid_size

    # Iterate through basis states and position circles on the grid
    for i, amp in enumerate(statevector):
        row = grid_size - (i // grid_size) - 1
        col = i % grid_size

        x = col * 2 * radius + radius  # Center x-coordinate of the circle
        y = row * 2 * radius + radius  # Center y-coordinate of the circle

        phase = np.angle(amp)
        hue = (phase + np.pi) / (2 * np.pi)  # Map phase to hue (0 to 1)
        color = hls_to_rgb(hue, 0.5, 1)

        circle = plt.Circle((x, y), np.abs(amp) * radius, color=color, alpha=0.5)
        ax.add_artist(circle)

        ax.text(x, y, f"{bin(i)[2:].zfill(num_qubits)}", ha='center', va='center', fontsize=6)

        # Set plot limits (adjust as needed)
    ax.set_xlim([0, grid_size * 2 * radius])
    ax.set_ylim([0, grid_size * 2 * radius])
    ax.set_title("Statevector Circle Notation")
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    plt.show()

# we need a way to parse the boolean expressions as LaTex strings
def tokenize(formula: str):
    # Replace LaTeX sequences with simple tokens and trim white spaces
    formula = repr(formula.replace(r'\land', '&').replace(r'\lor', '|')).replace(r'\neg', '~').replace(r" ", "")
    # Now, you can split by parentheses and operators. For a naive approach:
    tokens = re.findall(r'[A-Za-z]+|~[A-Za-z]+|[\(\)&|]', formula)
    return tokens

# Function to count unique uppercase letters
def count_unique_uppercase(tokens):
    # Extract all variable names (e.g., A, B, C) and store only their uppercase part
    variables = set(token for token in tokens if token.isalpha() and token.isupper())
    return len(variables), variables


formula = "(A \lor B) \land (\neg A \lor C) \land (\neg B \lor \neg C) \land (A \lor C)"

tokens = tokenize(formula)
token_string = " ".join(tokens)

# Count uppercase variables
unique_count, unique_variables = count_unique_uppercase(tokens)

sorted_variables = sorted(unique_variables)  # Ensures the order is consistent
var_dict = {var: idx for idx, var in enumerate(sorted_variables)}

was_cnf = is_cnf(token_string)

# if it is not a CNF convert it to a CNF with Sympy
if (not was_cnf):
    print("Not a CNF")
    token_string = to_cnf(token_string)

or_clauses = token_string.split("&")

# we count how many qubits we need. We sum the number of unique_count with the number of clauses
n = unique_count + len(or_clauses)

def get_repr(qc, is_inv, clause, i):
    clause = clause.replace("(", "").replace(")", "").replace(" ", "")
    sub_clause = clause.split("|")
    first = False
    second = False
    if len(sub_clause[0]) == 2: first = True # means that the formula is notted
    if len(sub_clause[1]) == 2: second = True # same as above

    # look up positions
    clause = clause.replace("~", "")
    sub_clause = clause.split("|")
    pos1 = var_dict[sub_clause[0]]
    pos2 = var_dict[sub_clause[1]]
    # put notted synbols
    if first: qc.x(pos1)
    if second: qc.x(pos2)

    qc.x(pos1)
    qc.x(pos2)

    if is_inv:
        qc.x(i)

    qc.mcx([pos1, pos2], i)

    if not is_inv:
        qc.x(i)

    qc.x(pos1)
    qc.x(pos2)

    if first: qc.x(pos1)
    if second: qc.x(pos2)
    qc.barrier()

def oracle(qc):
    for i, clause in enumerate(or_clauses):
        get_repr(qc, False, clause, unique_count + i)

    qc.mcp(np.pi, list(range(unique_count,n-1)), n-1)
    qc.barrier()

    reversed_or_clauses = or_clauses[::-1]

    for clause in reversed_or_clauses:
        get_repr(qc, True, clause, unique_count + i)
        i -= 1

def diffuser(qc, n):
    qc.h(range(n))
    qc.x(range(n))
    qc.mcp(np.pi, list(range(n-1)), n - 1)
    qc.x(range(n))
    qc.h(range(n))
    qc.barrier()

def create_circuit(qc):
    oracle(qc)
    diffuser(qc, unique_count)

qc = QuantumCircuit(n)
create_circuit(qc)

# now we want to repeat the oracle and the diffuser sqrt(2^n) times
reps = math.ceil( math.sqrt( 2**unique_count ))

qc = QuantumCircuit(n)
qc.h(list(range(unique_count)))
for i in range(reps):
    create_circuit(qc)

# create statevector
statevector = Statevector(qc)
# plot statevector
plot_statevector_circles(statevector)

qc.measure_all()
result = Sampler().run([qc], shots=100).result()[0]
co = result.data.meas.get_counts()

circuit_drawer(qc, output="mpl")

plot_histogram(co)
plt.show()

qc = RemoveBarriers()(qc)

print(qc.count_ops())

optimized_qc = transpile(qc, optimization_level=3)

print(optimized_qc.count_ops())

circuit_drawer(optimized_qc, output="mpl")
plt.show()