# basic imports
import numpy as np
import pydot
import matplotlib.pyplot as plt
import os

# causallearn imports
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils

# utils imports
from utils import (
    basic_causal_dataframe, 
    getCausalRelationship, 
    generate_graph_from_causes, 
    validate_all_solutions, 
    is_valid_o_to_solution
)

# SAT solvers
from SAT.classical import solveClassicalSAT
from SAT.fixedPointQuantum import solveFixedQuantunSAT

# --- Configuration ---
np.random.seed(0)
logging = True
# Create output directory if it doesn't exist
if not os.path.exists("output/FCI"):
    os.makedirs("output/FCI")

# --- Step 1: Run FCI to get a PAG ---
print("Step 1: Running FCI algorithm...")
data = basic_causal_dataframe()
variable_names = list(data.columns)
g, _ = fci(data.to_numpy(), node_names=variable_names)

# Save the initial FCI graph (PAG)
pdy = GraphUtils.to_pydot(g, labels=variable_names)
pdy.write_png("output/FCI/FCI_output.png")
print("FCI output graph saved to output/FCI/FCI_output.png")


# --- Step 2: Parse FCI Edges ---
print("\nStep 2: Parsing FCI edges into a structured list...")
edges = []
indices = np.where(g.graph != 0)
processed_pairs = set()

for i, j in zip(indices[0], indices[1]):
    node_pair = frozenset([i, j])
    if node_pair in processed_pairs:
        continue

    node1_name = g.nodes[i].get_name()
    node2_name = g.nodes[j].get_name()

    # Determine edge type based on causal-learn's internal representation
    if g.graph[i, j] == 1 and g.graph[j, i] == -1: # j -> i
        edges.append({'from': node2_name, 'to': node1_name, 'type': "->"})
    elif g.graph[i, j] == -1 and g.graph[j, i] == 1: # i -> j
        edges.append({'from': node1_name, 'to': node2_name, 'type': "->"})
    elif g.graph[i, j] == 1 and g.graph[j, i] == 2: # j o-> i
        edges.append({'from': node2_name, 'to': node1_name, 'type': "o->"})
    elif g.graph[i, j] == 2 and g.graph[j, i] == 1: # i o-> j
        edges.append({'from': node1_name, 'to': node2_name, 'type': "o->"})
    elif g.graph[i, j] == 2 and g.graph[j, i] == 2: # i o-o j
        edges.append({'from': node1_name, 'to': node2_name, 'type': "o-o"})
        processed_pairs.add(node_pair)
    elif g.graph[i, j] == 1 and g.graph[j, i] == 1: # i <-> j
        edges.append({'from': node1_name, 'to': node2_name, 'type': "<->"})
        processed_pairs.add(node_pair)

if logging: print(f"LOG: Parsed Edges: {edges}")


# --- Step 3: Map Causal Relationships to SAT Variables ---
print("\nStep 3: Mapping potential causal relationships to SAT variables...")
causal_dict = {}
for node1 in variable_names:
    for node2 in variable_names:
        if node1 == node2: continue
        # Variable for direct causation: node1 -> node2
        causal_dict[(node1, node2, 'direct')] = len(causal_dict) + 1
        # Variable for latent confounding: node1 <-> node2
        causal_dict[(node1, node2, 'latent')] = len(causal_dict) + 1

reversed_causal_dict = {v: k for k, v in causal_dict.items()}
if logging: print(f"LOG: Created {len(causal_dict)} SAT variables.")


# --- Step 4: Build SAT Clauses from FCI Edges ---
print("\nStep 4: Translating FCI edge constraints into CNF clauses...")
SATClauses = []
o_to_pairs = set()

for item in edges:
    f, t = item["from"], item["to"]
    
    # A -> B  (Direct edge)
    if item["type"] == "->":
        SATClauses.append([causal_dict[(f, t, "direct")]])
        SATClauses.append([-causal_dict[(t, f, "direct")]])
        SATClauses.append([-causal_dict[(f, t, "latent")]])

    # A o-> B (Not B -> A)
    elif item["type"] == "o->":
        SATClauses.append([-causal_dict[(t, f, 'direct')]])
        o_to_pairs.add((f, t)) # Constraint: No directed path from t to f

    # A o-o B (A->B or B->A or A<->B, but not A->B and B->A)
    elif item["type"] == "o-o":
        a = causal_dict[(f, t, 'direct')]
        b = causal_dict[(t, f, 'direct')]
        c = causal_dict[(f, t, 'latent')] # or (t, f, 'latent'), they are symmetric
        # Encodes (a OR b OR c) AND NOT (a AND b)
        SATClauses.append([a, b, c])
        SATClauses.append([-a, -b])

    # A <-> B (Latent confounder)
    elif item['type'] == "<->":
        SATClauses.append([-causal_dict[(f, t, 'direct')]])
        SATClauses.append([-causal_dict[(t, f, 'direct')]])
        SATClauses.append([causal_dict[(f, t, 'latent')]])

if logging: print(f"LOG: The SAT clauses are: {SATClauses}\n")

# --- Step 5: Remap variables for the solver ---
variable_set = {abs(var) for clause in SATClauses for var in clause}
cnf_variable_mapping = {var: i + 1 for i, var in enumerate(variable_set)}
reverse_cnf_variable_mapping = {v: k for k, v in cnf_variable_mapping.items()}

new_cnf = [[(cnf_variable_mapping[abs(var)] if var > 0 else -cnf_variable_mapping[abs(var)]) for var in clause] for clause in SATClauses]
if logging: print(f"LOG: The new CNF for solver is: {new_cnf}\n")

# --- Step 6: Solve SAT Problem (Classical and Quantum) ---
print("\nStep 6: Solving the SAT problem...")
# Solve classically to find one solution
is_sat_classical, classical_model_new = solveClassicalSAT(new_cnf)
if is_sat_classical:
    classical_model = [(reverse_cnf_variable_mapping[abs(item)] if item > 0 else -reverse_cnf_variable_mapping[abs(item)]) for item in classical_model_new]
    if logging: print(f"LOG: Classical SAT solver found a solution: {classical_model}\n")
else:
    print("LOG: Classical SAT solver returned UNSAT.")
    classical_model = []

# Solve with a "quantum" solver to find multiple solutions
is_sat_quantum, quantum_solutions_new = solveFixedQuantunSAT(new_cnf, 8, np.sqrt(0.1), debug=True, simulation=True)
if not is_sat_quantum:
    print("LOG: Quantum SAT solver returned UNSAT.")
    quantum_solutions_new = []


# --- Step 7: Process and Validate Solutions ---
print("\nStep 7: Validating and interpreting solutions...")

# Validate that solutions satisfy the boolean clauses
validity = validate_all_solutions(new_cnf, quantum_solutions_new)
valid_quantum_solutions = [sol for sol, v in zip(quantum_solutions_new, validity) if v]
print(f"\033[1m\033[4mLOG: Found {sum(validity)}/{len(validity)} valid solutions from quantum solver.\033[0m\n")

# Map solutions back to original variable IDs
mapped_solutions = []
for solution in valid_quantum_solutions:
    mapped_solution = [(reverse_cnf_variable_mapping[abs(item)] if item > 0 else -reverse_cnf_variable_mapping[abs(item)]) for item in solution]
    mapped_solutions.append(mapped_solution)

# Filter solutions based on graph constraints (e.g., o-> which implies acyclicity subset)
before_count = len(mapped_solutions)
final_solutions = [sol for sol in mapped_solutions if is_valid_o_to_solution(sol, reversed_causal_dict, o_to_pairs, variable_names)]
after_count = len(final_solutions)
if logging:
    print(f"\033[1m\033[4mLOG: Filtered out {before_count - after_count} solutions violating o-> constraints.\033[0m")
    print(f"\033[1m\033[4mLOG: Remaining valid solutions: {after_count}\033[0m\n")


# --- Step 8: Visualize the Results ---
print("\nStep 8: Generating output graphs...")

# Generate and save classical solution graph
if classical_model:
    classical_causes = getCausalRelationship(classical_model, reversed_causal_dict)
    generate_graph_from_causes(classical_causes, variable_names, "output/FCI/classical_output.png")

# --- NEW: Visualize all valid quantum solutions ---
def visualize_all_solutions(solutions, output_dir_base, reversed_causal_dict, all_nodes, logging=True):
    """
    Iterates through SAT solutions, generates a causal graph for each,
    and saves it as a PNG image.
    """
    output_dir = os.path.join(output_dir_base, "quantum_solutions")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not solutions:
        if logging: print("\n\033[1m\033[4mLOG: No valid solutions to visualize.\033[0m")
        return

    if logging: print(f"\n\033[1m\033[4mLOG: Generating graphs for {len(solutions)} valid solutions...\033[0m")

    for i, solution in enumerate(solutions):
        graph = pydot.Dot(graph_type='digraph', rankdir='LR', label=f"Solution {i+1}")
        
        # Add all nodes
        for node_name in all_nodes:
            graph.add_node(pydot.Node(node_name, shape='circle', style='filled', fillcolor='lightblue'))
            
        # Add edges based on the solution
        for var in solution:
            if var > 0: # Relationship is true
                original_var_id = abs(var)
                from_node, to_node, edge_type = reversed_causal_dict[original_var_id]
                
                if edge_type == 'direct':
                    edge = pydot.Edge(from_node, to_node, dir='forward', color='black', penwidth=1.5)
                    graph.add_edge(edge)
                elif edge_type == 'latent':
                    edge = pydot.Edge(from_node, to_node, dir='both', color='red', style='dashed', penwidth=1.5)
                    graph.add_edge(edge)

        filename = os.path.join(output_dir, f"solution_{i+1}.png")
        try:
            graph.write_png(filename)
            if logging: print(f"LOG: Saved solution graph to {filename}")
        except Exception as e:
            print(f"Error saving graph {filename}: {e}")
            
    if logging: print(f"\n\033[1m\033[4mLOG: All solution graphs saved in '{output_dir}' directory.\033[0m")

# Generate visualization of quantum solutions
visualize_all_solutions(final_solutions, "output/FCI", reversed_causal_dict, variable_names, logging=logging)
