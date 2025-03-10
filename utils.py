import pandas as pd
import numpy as np
from causallearn.graph.Edge import Edge
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import pydot

# Initialize a counter for variable IDs
next_var_id = 1

# create causal dataframe
def basic_causal_dataframe() -> pd.DataFrame:
    X = np.random.uniform(size=1000)
    eps = np.random.normal(size=1000)
    eps = np.zeros(1000)
    delta = np.random.uniform(size=1000)
    delta = np.zeros(1000)
    Y = -7 * X + 0.5 * delta
    Z = 2 * X + Y + eps

    # Create DataFrame with named variables
    return pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

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
    
    if mean_1 > mean_2:
        return cluster_1
    elif mean_2 > mean_1:  
        return cluster_2
    else:
        raise Exception("Something went really wrong with the clustering of the solutions.... the means are equal")
    
def elbow_plot(counts: dict, cutoff):
    # take all the values from the dictionary, and place them into a numpy array
    values = np.array(list(counts.values()))
    cutoff = np.array(list(cutoff.values()))
    
    values.sort()
    
    plt.plot(values)
    plt.axvline(x=len(values) - len(cutoff), color='r', linestyle='--')
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
        elif rel["edge"] == "latent" and rel["exists"]:
            graph.add_edge(pydot.Edge(rel["node1"], rel["node2"], style="dotted", arrowhead="none"))
    
    return graph

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