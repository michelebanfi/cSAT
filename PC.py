# basic imports
import numpy as np
import pydot
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
import os

# causallearn imports
from causallearn.search.ConstraintBased.PC import pc

# utils imports
from utils import basic_causal_dataframe, getCausalRelationship, generate_graph_from_causes, visualize_quantum_solutions

# SAT solvers
from SAT.classical import solveClassicalSAT
from SAT.quantum import solveQuantumSAT
from SAT.validateSolution import validate_all_solutions

# set the seed for reproducibility
np.random.seed(0)

logging = True

# create the causal dataframe
data = basic_causal_dataframe()

# extract the variable names
variable_names = list(data.columns)

if logging: print("\nLOG: Running the PC algorithm on the following data\n")

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

if logging:  print(f"LOG: The extracted edges are: {edges}\n")

# create a list of possible causal relationship in the variables
causal_dict = {}
for node1 in node_mapping:
    for node2 in node_mapping:
        for edge in ['direct']:
            causal_dict[(node1, node2, edge)] = len(causal_dict) + 1
            
reversed_causal_dict = {v: k for k, v in causal_dict.items()}
            
# print(f"LOG: The causal dictionary is: {causal_dict}\n")

# now we need to create the SAT clauses
SATClauses = []

for item in edges:
    if item['type'] == '->':
        # there MUST be a direct edge from node1 to node2 and NO direct edge from node2 to node1
        SATClauses.append([causal_dict[(item['from'], item['to'], 'direct')]])
        SATClauses.append([-causal_dict[(item['to'], item['from'], 'direct')]])
    elif item['type'] == '--':
        # there MUST be exactly one direct edge: either from node1 to node2 OR from node2 to node1 (XOR)
        # For XOR in CNF: (A OR B) AND (NOT A OR NOT B)
        SATClauses.append([causal_dict[(item['from'], item['to'], 'direct')], causal_dict[(item['to'], item['from'], 'direct')]])
        SATClauses.append([-causal_dict[(item['from'], item['to'], 'direct')], -causal_dict[(item['to'], item['from'], 'direct')]])
    elif item['type'] == '<->':
        # there MUST be a direct edge from node1 to node2 and a direct edge from node2 to node1
        SATClauses.append([causal_dict[(item['from'], item['to'], 'direct')]])
        SATClauses.append([causal_dict[(item['to'], item['from'], 'direct')]])
        
        
# print(f"LOG: The SAT clauses are: {SATClauses}\n")

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

# print(f"LOG: The variable mapping is: {cnf_variable_mapping}\n")
# print(f"LOG: The reverse variable mapping is: {reverse_cnf_variable_mapping}\n")

# so the new cnf will be
new_cnf = []
for clause in SATClauses:
    new_clause = []
    for var in clause:
        new_var = cnf_variable_mapping[abs(var)]
        new_clause.append(new_var if var > 0 else -new_var)
    new_cnf.append(new_clause)
    
if logging: print(f"LOG: The new CNF is: {new_cnf}\n")
# new_cnf = [[1, -1], [2, -2], [3, -3], [4, -4]]


# solve the classical SAT
is_sat, model = solveClassicalSAT(new_cnf)


# just to map back the model
temp = []
for item in model:
    temp.append(reverse_cnf_variable_mapping[abs(item)] if item > 0 else -reverse_cnf_variable_mapping[abs(item)])
classical_model = temp

# output the results:
if logging: print(f"LOG: Classical SAT solver returned: {is_sat}\n")
if logging: print(f"LOG: The model is: {classical_model}\n")

# Get solutions from quantum SAT solver
is_sat, quantum_solutions = solveQuantumSAT(new_cnf)

# print(f"DEBUG: Quantum SAT solver returned: {quantum_solutions}\n")

# print(reverse_cnf_variable_mapping)

# check for quantum solutions validity
if is_sat:
    
    # Validate all solutions, which is an array of boolean values
    validity = validate_all_solutions(new_cnf, quantum_solutions)
    
    # count the number of valid solutions
    valid_count = sum(validity)
    print(f"\033[1m\033[4mLOG: The number of valid quantum solutions is: {valid_count} out of {len(quantum_solutions    )}\033[0m\n")
    
    # Filter out only the valid solutions
    quantum_solutions = [solution for solution, valid in zip(quantum_solutions, validity) if valid]

# Map back all solutions using reverse_cnf_variable_mapping
mapped_solutions = []
for solution in quantum_solutions:
    mapped_solution = []
    for item in solution:
        # print(item)
        mapped_var = reverse_cnf_variable_mapping[abs(item)]
        mapped_solution.append(mapped_var if item > 0 else -mapped_var)
    mapped_solutions.append(mapped_solution)

if logging: print(f"LOG: Quantum SAT solver returned: {is_sat}\n")
if logging: print(f"LOG: The models are: {mapped_solutions}\n")

# check if quantum does indeed contain the classical solution
if classical_model in mapped_solutions:
    print(f"\033[1m\033[4mLOG: The classical solution is in the quantum solutions!\033[0m\n")
else:
    if logging: print(f"LOG: The classical solution is NOT in the quantum solutions\n")
    
# get classical direct cause
classical_direct_causes = [rel for rel in getCausalRelationship(classical_model, reversed_causal_dict) if rel["edge"] == "direct" and rel["exists"]]

# Generate and save classical solution
classical_graph = generate_graph_from_causes(classical_direct_causes)
classical_graph.write_png("output/PC/classical_output.png")

# Generate visualization of quantum solutions
if mapped_solutions:
    visualize_quantum_solutions(mapped_solutions, "output/PC", reversed_causal_dict, logging=logging)

# After defining generate_graph_from_causes function, add this to visualize the PC output directly
def visualize_pc_output():
    # Create a graph to visualize the PC algorithm output
    pc_graph = pydot.Dot("pc_graph", graph_type="digraph")
    
    # Add nodes
    for node_name in node_mapping:
        pc_graph.add_node(pydot.Node(node_name))
    
    # Add edges based on the extracted edges
    for edge in edges:
        if edge['type'] == '->':
            pc_graph.add_edge(pydot.Edge(edge['from'], edge['to']))
        elif edge['type'] == '--':
            pc_graph.add_edge(pydot.Edge(edge['from'], edge['to'], dir="none"))
        elif edge['type'] == '<->':
            pc_graph.add_edge(pydot.Edge(edge['from'], edge['to'], dir="both"))
    
    # Save the graph
    pc_graph.write_png("output/PC/PC_output.png")
    if logging: print("LOG: Saved PC algorithm output visualization\n")

# Generate the PC algorithm output visualization
visualize_pc_output()

