# basic imports
import numpy as np
import pydot
import matplotlib.pyplot as plt
import os

# causallearn imports
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils

# utils imports
from utils import basic_causal_dataframe

# SAT solvers
from SAT.classical import solveClassicalSAT
from SAT.quantum import solveQuantumSAT
from SAT.validateSolution import validate_all_solutions


np.random.seed(0)

logging = True

data = basic_causal_dataframe()

# extract the variable names
variable_names = list(data.columns)

g, edges = fci(data.to_numpy(), node_names=variable_names)

pdy = GraphUtils.to_pydot(g)
pdy.write_png("output/FCI/FCI_output.png")

edges = []
indices = np.where(g.graph != 0)
processed_pairs = set()

for i, j in zip(indices[0], indices[1]):
    node_pair = frozenset([i.item(), j.item()])
    
    if node_pair in processed_pairs:
        continue
    
    if g.graph[i, j] == -1 and g.graph[j, i] == 1:
        edges.append({
            'from': g.nodes[i].get_name(),
            'to': g.nodes[j].get_name(),
            'type': "->"
        })
    elif g.graph[i, j] == 2 and g.graph[j, i] == 1:
        edges.append({
            'from': g.nodes[j].get_name(),
            'to': g.nodes[i].get_name(),
            'type': "o->"
        })
    elif g.graph[i, j] == 2 and g.graph[j, i] == 2:
        edges.append({
            'from': g.nodes[i].get_name(),
            'to': g.nodes[j].get_name(),
            'type': "o-o"
        })
        processed_pairs.add(node_pair)
    elif g.graph[i, j] == 1 and g.graph[j, i] == 1:
        edges.append({
            'from': g.nodes[i].get_name(),
            'to': g.nodes[j].get_name(),
            'type': "<->"
        })
        processed_pairs.add(node_pair)

# create a list of possible causal relationship in the variables
causal_dict = {}
for node1 in variable_names:
    for node2 in variable_names:
        for edge in ['direct', 'latent']:
            causal_dict[(node1, node2, edge)] = len(causal_dict) + 1

reversed_causal_dict = {v: k for k, v in causal_dict.items()}

SATClauses = []

for item in edges:
    if item["type"] == "->":
        SATClauses.append([causal_dict[(item["from"], item["to"], "direct")]])
        SATClauses.append([-causal_dict[(item["to"], item["from"], "direct")]])
    elif item["type"] == "o->":
        SATClauses.append([-causal_dict[(item['to'], item['from'], 'direct')]])
    elif item["type"] == "o-o":
        
        # (a ∨ b ∨ c ∨ (a ∧ c) ∨ (b ∧ c)) ∧ ¬(a ∧ b)
        # a = A cause B
        # b = B cause A
        # c = A cause B latent
        
        # the CNF is (a∨b∨c) ∧ (¬a∨¬b∨c) ∧ (¬a∨¬b∨¬c)
        SATClauses.append([
            causal_dict[(item['from'], item['to'], 'direct')], 
            causal_dict[(item['to'], item['from'], 'direct')], 
            causal_dict[(item['from'], item['to'], 'latent')]
        ])
        SATClauses.append([
            -causal_dict[(item['from'], item['to'], 'direct')], 
            -causal_dict[(item['to'], item['from'], 'direct')], 
            causal_dict[(item['from'], item['to'], 'latent')]
        ])
        SATClauses.append([
            -causal_dict[(item['from'], item['to'], 'direct')], 
            -causal_dict[(item['to'], item['from'], 'direct')], 
            -causal_dict[(item['from'], item['to'], 'latent')]
        ])
            
    elif item['type'] == "<->":
        SATClauses.append([-causal_dict[(item['from'], item['to'], 'direct')]])
        SATClauses.append([-causal_dict[(item['to'], item['from'], 'direct')]])
        SATClauses.append([causal_dict[(item['from'], item['to'], 'latent')]])
    
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

# print(f"LOG: The variable mapping is: {cnf_variable_mapping}\n")
# print(f"LOG: The reverse variable mapping is: {reverse_cnf_variable_mapping}\n")

new_cnf = []
for clause in SATClauses:
    new_clause = []
    for var in clause:
        new_var = cnf_variable_mapping[abs(var)]
        new_clause.append(new_var if var > 0 else -new_var)
    new_cnf.append(new_clause)
    
if logging: print(f"LOG: The new CNF is: {new_cnf}\n")

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
    
def getCausalRelationship(model):
    causal_relationship = []
    for item in model:
        absolute_item = abs(item)
        if absolute_item in reversed_causal_dict:
            node1, node2, edge = reversed_causal_dict[absolute_item]
            causal_relationship.append({
                "node1": node1,
                "node2": node2,
                "edge": edge,
                "exists": True if item > 0 else False    
            })
    return causal_relationship

# Extract graph generation into a reusable function
def generate_graph_from_causes(direct_causes):
    nodes = set()
    for rel in direct_causes:
        nodes.add(rel["node1"])
        nodes.add(rel["node2"])
    graph = pydot.Dot("my_graph", graph_type="digraph")
    for node in nodes:
        graph.add_node(pydot.Node(node))

    for rel in direct_causes:
        if rel["edge"] == "direct" and rel["exists"]:
            graph.add_edge(pydot.Edge(rel["node1"], rel["node2"]))
        elif rel["edge"] == "latent" and rel["exists"]:
            graph.add_edge(pydot.Edge(rel["node1"], rel["node2"], style="dotted", arrowhead="none"))
    
    return graph

# get classical direct cause
classical_direct_causes = [rel for rel in getCausalRelationship(classical_model) if rel["edge"] == "direct" and rel["exists"]]

# Generate and save classical solution
classical_graph = generate_graph_from_causes(classical_direct_causes)
classical_graph.write_png("output/FCI/classical_FCI_output.png")

# Process quantum solutions and create visualization grid
def visualize_quantum_solutions(mapped_solutions, max_solutions=10):
    # Limit to maximum number of solutions
    solutions_to_show = min(len(mapped_solutions), max_solutions)
    solutions = mapped_solutions[:solutions_to_show]
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(solutions_to_show)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    # Make axes a 2D array even if it's a single value
    if solutions_to_show == 1:
        axes = np.array([[axes]])
    elif grid_size == 1:
        axes = axes.reshape(1, -1)
    
    # Keep track of temporary files for later cleanup
    temp_files = []
    
    # Generate graph for each solution
    for i, solution in enumerate(solutions):
        if i >= solutions_to_show:
            break
            
        row = i // grid_size
        col = i % grid_size
        
        # Get direct causes for this solution
        quantum_direct_causes = [rel for rel in getCausalRelationship(solution) if rel["exists"]]
        
        # Generate graph
        graph = generate_graph_from_causes(quantum_direct_causes)
        
        # Save to temporary file
        temp_filename = f"output/FCI/temp_quantum_solution_{i}.png"
        graph.write_png(temp_filename)
        temp_files.append(temp_filename)
        
        # Display in the subplot
        img = plt.imread(temp_filename)
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Solution {i+1}")
        axes[row, col].axis('off')
    
    # Hide any unused subplots
    for i in range(solutions_to_show, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig("output/FCI/quantum_FCI_outputs.png", dpi=300)
    plt.close(fig)
    
    # Clean up temporary files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            if logging: print(f"Removed temporary file: {temp_file}")
        except Exception as e:
            if logging: print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    
    if logging: print(f"LOG: Generated visualization of {solutions_to_show} quantum solutions\n")

# Generate visualization of quantum solutions
if mapped_solutions:
    visualize_quantum_solutions(mapped_solutions)
