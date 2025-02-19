# standard imports
import numpy as np
import pandas as pd
import pydot

# causal-learn imports
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils

# pysat imports
from pysat.formula import CNF
from pysat.solvers import Glucose3

# utils
from utils import basic_causal_dataframe, get_edge, create_variable_mapping, add_edge_constraints, add_transitive_closure_constraints, add_no_ancestor_constraints

data = basic_causal_dataframe()

variable_names = list(data.columns)

g, edges = fci(data.to_numpy(), alpha=0.05)

sat_clauses = []
formatted_edges = []
for edge in edges:
    formatted_edges.append((edge.node1.name, edge.node2.name, get_edge(edge)))

var_mapping = create_variable_mapping({}, variable_names)

cnf = add_edge_constraints(formatted_edges, variable_names, var_mapping)

# Add transitive closure constraints
cnf.extend(add_transitive_closure_constraints(variable_names, var_mapping))

# Add the no-ancestor constraints
cnf.extend(add_no_ancestor_constraints(formatted_edges, var_mapping))

# iterate through the clauses and count the number of variables
variable_set = set()
for clause in cnf:
    for var in clause:
        variable_set.add(abs(var))

new_var = list(range(1, len(variable_set) + 1))

# create a mapping from old variable to new variable
cnf_variable_mapping = {}
for i, var in enumerate(variable_set):
    cnf_variable_mapping[var] = new_var[i]

new_cnf = []
for clause in cnf:
    new_clause = []
    for var in clause:
        new_var = cnf_variable_mapping[abs(var)]
        new_clause.append(new_var if var > 0 else -new_var)
    new_cnf.append(new_clause)

# create the formula as CNF
formula = CNF(from_clauses=new_cnf)

solver = Glucose3()
solver.append_formula(formula)

is_sat = solver.solve()

print("The problem has a solution") if is_sat else print("The problem has no solution")

model = solver.get_model()

# map back with cnf_variable_mapping
# reverse mapping
reverse_cnf_variable_mapping = {v: k for k, v in cnf_variable_mapping.items()}

temp = []
for item in model:
    temp.append(reverse_cnf_variable_mapping[abs(item)] if item > 0 else -reverse_cnf_variable_mapping[abs(item)])
model = temp

# Create reverse mapping for interpretation
reverse_mapping = {v: k for k, v in var_mapping.items()}

causal_relationship = []

for item in model:
    absolute_value = abs(item)
    if absolute_value in reverse_mapping:
        node1, node2, edge = reverse_mapping[absolute_value]
        causal_relationship.append({
            "node1": node1,
            "node2": node2,
            "edge": edge,
            "exists": True if item > 0 else False
        })

# de-allocate solver
solver.delete()

direct_causes = [rel for rel in causal_relationship if rel["edge"] == "direct" and rel["exists"]]
latent_causes = [rel for rel in causal_relationship if rel["edge"] == "latent" and rel["exists"]]

# create a set of variables which will be the nodes
nodes = set()
for rel in causal_relationship:
    nodes.add(rel["node1"])
    nodes.add(rel["node2"])
graph = pydot.Dot("my_graph", graph_type="digraph")
for node in nodes:
    graph.add_node(pydot.Node(node))

for rel in causal_relationship:
    if rel["edge"] == "direct" and rel["exists"]:
        graph.add_edge(pydot.Edge(rel["node1"], rel["node2"]))
    elif rel["edge"] == "latent" and rel["exists"]:
        graph.add_edge(pydot.Edge(rel["node2"], rel["node1"], style="dotted"))

graph.write_png("output/output.png")