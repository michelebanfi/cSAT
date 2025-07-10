import pandas as pd
import numpy as np
from causallearn.graph.Edge import Edge
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import pydot
from SAT.validateSolution import has_path

# Initialize a counter for variable IDs
next_var_id = 1

# create causal dataframe
def basic_causal_dataframe() -> pd.DataFrame:
    np.random.seed(42)  # Set seed for reproducibility
    X = np.random.uniform(size=1000)
    eps_Y = np.random.normal(0, 0.1, size=1000)  # Add noise to Y
    eps_Z = np.random.normal(0, 0.1, size=1000)  # Add noise to Z
    eps_W = np.random.normal(0, 0.1, size=1000)  # Add noise to W
    delta = np.random.uniform(size=1000)
    
    Y = -7 * X + 0.5 * delta + eps_Y
    Z = 2 * X + Y + eps_Z
    W = 3 * X + 2 * Y + eps_W

    # Y = X + eps_Y
    # Z = X + Y + eps_Z
    # W = 3 * X + 2 * Y + eps_W

    # Create DataFrame with named variables
    return pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

# def basic_causal_dataframe():
#     """Creates a sample dataframe with known causal structure for testing."""
#     np.random.seed(42)
#     # Structure: Z -> X, Z -> Y, L -> X, L -> Y (L is latent)
#     # This should result in X o-o Y from FCI
#     size = 500
#     Z = np.random.randn(size, 1)
#     L = np.random.randn(size, 1) # Latent variable
#     X = 0.8 * Z + 0.7 * L + np.random.randn(size, 1) * 0.3
#     Y = 0.6 * Z - 0.5 * L + np.random.randn(size, 1) * 0.3
#     A = 0.9 * X + np.random.randn(size, 1) * 0.2 # X -> A
#     B = 0.7 * Y + np.random.randn(size, 1) * 0.2 # Y -> B

#     data = pd.DataFrame(np.hstack([X, Y, Z, A]), columns=['X', 'Y', 'Z', 'A'])
#     return data

def cluster_solutions(count: dict):
    # create an array of probabilites maining the same order
    probabilities = []
    for key in count.keys():
        probabilities.append(count[key])
        
    # transform the array into a numpy array and reshape for k-means
    # K-means requires 2D array, even for 1D data, so reshape (-1, 1)
    probabilities = np.array(probabilities).reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(probabilities)
    
    kmeans_labels = kmeans.labels_
    
    # now we need to split the counts into two clusters, which are denoted by the labels
    cluster_1 = {}
    cluster_2 = {}
    for idx, key in enumerate(count.keys()):
        if kmeans_labels[idx] == 0:
            cluster_1[key] = count[key]
        else:
            cluster_2[key] = count[key]
            
    # now we need to return the take the clusters which has the mean value of the probabilities higher
    # than the other cluster
    mean_1 = np.mean(list(cluster_1.values()))
    mean_2 = np.mean(list(cluster_2.values()))
    
    # calculate the Silhouette score using scikit
    silhouette_avg = silhouette_score(probabilities, kmeans_labels)
    
    if mean_1 > mean_2:
        return cluster_1, silhouette_avg
    elif mean_2 > mean_1:  
        return cluster_2, silhouette_avg
    else:
        raise Exception("Something went really wrong with the clustering of the solutions.... the means are equal")
    
def elbow_plot(counts: dict, cutoff):
    # take all the values from the dictionary, and place them into a numpy array
    values = np.array(list(counts.values()))
    cutoff = np.array(list(cutoff.values()))
    
    values.sort()
    
    for i in range(len(values)):
        print(f"({i}, {values[i]}),", end=' ')
    
    print(len(values) - len(cutoff))
    
    plt.scatter(x=list(range(len(values))), y=values)
    plt.axvline(x=len(values) - len(cutoff), color='r', linestyle='--')
    plt.title("Outcome probabilities")
    plt.xlabel("Outcome")
    plt.ylabel("Probability")
    plt.xticks([])
    plt.savefig("debug/elbow_plot.png")
    plt.close()
    
def structural_check(cnf: list):
    for item in cnf:
        variables = set()
        for var in item:
            variables.add(abs(var))
        if len(item) != len(variables):
            raise Exception("There is something wrong with yout variables. Two variables are the same in a clause")

def getCausalRelationship(model, reversed_causal_dict):
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
        # elif rel["edge"] == "latent" and rel["exists"]:
        #     graph.add_edge(pydot.Edge(rel["node1"], rel["node2"], style="dotted", arrowhead="none"))
    
    return graph

# def generate_graph_from_causes(causes, all_nodes, filename):
#     """Generates a pydot graph from a list of causal relationships (for the classical solution)."""
#     graph = pydot.Dot(graph_type='digraph', rankdir='LR', label="Classical Solution")
    
#     for node_name in all_nodes:
#         graph.add_node(pydot.Node(node_name)) # , shape='circle', style='filled', fillcolor='lightgreen'
        
#     # print(causes)
        
#     for cause in causes:
#         if cause['exists']:
#             if cause['edge'] == 'direct':
#                 edge = pydot.Edge(cause['node1'], cause['node2'], dir='forward', color='black', penwidth=1.5)
#                 graph.add_edge(edge)
#             elif cause['edge'] == 'latent':
#                 edge = pydot.Edge(cause['node1'], cause['node2'], dir='both', color='red', style='dashed', penwidth=1.5)
#                 graph.add_edge(edge)
    
#     try:
#         graph.write_png(filename)
#         print(f"LOG: Saved classical graph to {filename}")
#     except Exception as e:
#         print(f"Error saving graph {filename}: {e}")
#     return graph

def visualize_quantum_solutions(mapped_solutions, output_dir, reversed_causal_dict, max_solutions=10, logging=True):
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
        quantum_direct_causes = [rel for rel in getCausalRelationship(solution, reversed_causal_dict) 
                               if rel["exists"]]
        
        # Generate graph
        graph = generate_graph_from_causes(quantum_direct_causes)
        
        # Save to temporary file
        temp_filename = f"{output_dir}/temp_quantum_solution_{i}.png"
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
    plt.savefig(f"{output_dir}/quantum_outputs.png", dpi=300)
    plt.close(fig)
    
    # Clean up temporary files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            if logging: print(f"Removed temporary file: {temp_file}")
        except Exception as e:
            if logging: print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    
    if logging: print(f"LOG: Generated visualization of {solutions_to_show} quantum solutions\n")
    
def is_valid_o_to_solution(solution, reversed_causal_dict, o_to_pairs, all_nodes):
    """
    Checks if a given SAT solution respects the o-> constraint (no path from 'to' to 'from').
    This is a critical step for ensuring the resulting graphs are valid MAGs.
    """
    # Build a directed graph from the current solution for path checking
    adj = {node: [] for node in all_nodes}
    for var in solution:
        if var > 0:
            from_node, to_node, edge_type = reversed_causal_dict[abs(var)]
            if edge_type == 'direct':
                adj[from_node].append(to_node)

    # For each o-> constraint (e.g., F o-> T), check for a forbidden path from T to F
    for f_node, t_node in o_to_pairs:
        if has_path(adj, t_node, f_node):
            return False # This solution is invalid because it implies T is an ancestor of F
            
    return True # All o-> constraints are respected

def validate_all_solutions(cnf, solutions):
    """Validates a list of solutions against a CNF formula."""
    validity = []
    for sol in solutions:
        is_valid = True
        solution_set = set(sol)
        for clause in cnf:
            if not any(literal in solution_set for literal in clause):
                is_valid = False
                break
        validity.append(is_valid)
    return validity