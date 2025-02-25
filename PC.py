# basic imports
import pandas as pd
import numpy as np
import pydot
import math
import re
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb

# causallearn imports
from causallearn.search.ConstraintBased.PC import pc

# pysat imports
from pysat.formula import CNF
from pysat.solvers import Glucose3

# sympy imports
from sympy.logic.boolalg import is_cnf, to_cnf

# qiskit imports
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import circuit_drawer
from qiskit import transpile
from qiskit.transpiler.passes import RemoveBarriers

# utils imports
from utils import basic_causal_dataframe

# SAT solvers
from SAT.classical import solveClassicalSAT


# create the causal dataframe
data = basic_causal_dataframe()

# extract the variable names
variable_names = list(data.columns)

print("\nLOG: Running the PC algorithm on the following data\n")

# run the PC algorithm
g = pc(data.to_numpy(), show_progress=False)

# rename the nodes
for i, node in enumerate(g.G.nodes):
    node.name = variable_names[i]
    
# save the node mapping and the reversed mapping
node_mapping = {node.name: index for index, node in enumerate(g.G.nodes)}
reversed_node_mapping = {index: node.name for index, node in enumerate(g.G.nodes)}

# Now we need to extract the edges:
edges = []
indices = np.where(g.G.graph != 0)
processed_pairs = set()

for i, j in zip(indices[0], indices[1]):
    
    node_pair = frozenset([i.item(), j.item()])
    
    if node_pair in processed_pairs:
        continue
        
    if g.G.graph[i,j] == 1 and g.G.graph[j,i] == -1:
        edges.append({
            'from': reversed_node_mapping[i.item()],
            'to': reversed_node_mapping[j.item()],
            'type': "->"
        })
    
    elif g.G.graph[i,j] == -1 and g.G.graph[j,i] == -1:
        edges.append({
            'from': reversed_node_mapping[i.item()],
            'to': reversed_node_mapping[j.item()],
            'type': "--"
        })
        processed_pairs.add(node_pair) 
    
    elif g.G.graph[i,j] == 1 and g.G.graph[j,i] == 1:
        edges.append({
            'from': reversed_node_mapping[i.item()],
            'to': reversed_node_mapping[j.item()],
            'type': "<->"
        })
        processed_pairs.add(node_pair)

print(f"LOG: The extracted edges are: {edges}\n")

# create a list of possible causal relationship in the variables
causal_dict = {}
for node1 in node_mapping:
    for node2 in node_mapping:
        for edge in ['direct']:
            causal_dict[(node1, node2, edge)] = len(causal_dict) + 1
            
print(f"LOG: The causal dictionary is: {causal_dict}\n")

# now we need to create the SAT clauses
SATClauses = []
index = 0

for item in edges:
    if item['type'] == '->':
        # there MUST be a direct edge from node1 to node2 and NO direct edge from node2 to node1
        SATClauses.append([causal_dict[(item['from'], item['to'], 'direct')]])
        SATClauses.append([-causal_dict[(item['to'], item['from'], 'direct')]])
    elif item['type'] == '--':
        # there MUST be a direct edge from node1 to node2 OR a direct edge from node2 to node1
        SATClauses.append([causal_dict[(item['from'], item['to'], 'direct')], causal_dict[(item['to'], item['from'], 'direct')]])
    elif item['type'] == '<->':
        # there MUSTN'T be a direct edge from node1 to node2 AND a direct edge from node2 to node1
        SATClauses.append([-causal_dict[(item['from'], item['to'], 'direct')]])
        SATClauses.append([-causal_dict[(item['to'], item['from'], 'direct')]])
        
        
print(f"LOG: The SAT clauses are: {SATClauses}\n")

# iterate through the clauses and count the number of variables
variable_set = set()
for clause in SATClauses:
    for var in clause:
        variable_set.add(abs(var))
        
new_var = list(range(1, len(variable_set) + 1))

# create a mapping from old variable to new variable
cnf_variable_mapping = {}
for i, var in enumerate(variable_set):
    cnf_variable_mapping[var] = new_var[i]
    
# reverse the mapping
reverse_cnf_variable_mapping = {v: k for k, v in cnf_variable_mapping.items()}

print(f"LOG: The variable mapping is: {cnf_variable_mapping}\n")
print(f"LOG: The reverse variable mapping is: {reverse_cnf_variable_mapping}\n")

# so the new cnf will be
new_cnf = []
for clause in SATClauses:
    new_clause = []
    for var in clause:
        new_var = cnf_variable_mapping[abs(var)]
        new_clause.append(new_var if var > 0 else -new_var)
    new_cnf.append(new_clause)
    
print(f"LOG: The new CNF is: {new_cnf}\n")

# solve the classical SAT
is_sat, model = solveClassicalSAT(new_cnf)

# just to map back the model
temp = []
for item in model:
    temp.append(reverse_cnf_variable_mapping[abs(item)] if item > 0 else -reverse_cnf_variable_mapping[abs(item)])
model = temp

# solve with the quantum version


# output the results:
print(f"LOG: Classical SAT solver returned: {is_sat}\n")
print(f"LOG: The model is: {model}\n")