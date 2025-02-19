import pandas as pd
import numpy as np
from causallearn.graph.Edge import Edge

# Initialize a counter for variable IDs
next_var_id = 1

# create causal dataframe
def basic_causal_dataframe() -> pd.DataFrame:
    X = np.random.uniform(size=1000)
    eps = np.random.normal(size=1000)
    delta = np.random.uniform(size=1000)
    Y = -7 * X + 0.5 * delta
    Z = 2 * X + Y + eps

    # Create DataFrame with named variables
    return pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

# translate the edges to a more readable format
def get_endpoint_type(endpoint: int, isFirst: bool):
    if endpoint == -1:
        return "-"
    elif endpoint == 1:
        return "<" if isFirst else ">"
    elif endpoint == 2:
        return "o"

def get_edge(edge: Edge):
    start = edge.numerical_endpoint_1
    end = edge.numerical_endpoint_2

    return f"{get_endpoint_type(start, True)}-{get_endpoint_type(end, False)}"

def create_variable_mapping(var_mapping, nodes):
    for n1 in nodes:
        for n2 in nodes:
            for edge_type in ["direct", "latent", "transitive"]:
                var_mapping[(n1, n2, edge_type)] = len(var_mapping) + 1
    return var_mapping

def get_next_var_id():
    global next_var_id
    var_id = next_var_id
    next_var_id += 1
    return var_id

# create the CNF clauses for the edge constraints
def add_edge_constraints(edges, all_nodes, var_mapping):
    cnf = []
    for n1, n2, edge_type in edges:
            if edge_type == '-->': # A is a direct cause of B

                # Direct causation must be true
                cnf.append([var_mapping[(n1, n2, 'direct')]])

                # No latent common cause
                cnf.append([-var_mapping[(n1, n2, 'latent')]])

            elif edge_type == 'o->': # B is not an ancestor of A

                cnf.append([-var_mapping[(n2, n1, 'direct')]])

                # For ancestral relationships, we need to prevent all paths from B to A
                # This requires additional variables to represent transitive relationships
                for intermediate in all_nodes:
                    if intermediate != n1 and intermediate != n2:
                        # If B→C and C→A, then B is an ancestor of A, which is prohibited
                        cnf.append([
                            -var_mapping[(n2, intermediate, 'direct')],
                            -var_mapping[(intermediate, n1, 'direct')]
                        ])

                        # For longer paths, we would need to recursively consider all possible paths
                        # This is complicated in pure SAT, but can be handled more easily with auxiliary variables

            elif edge_type == 'o-o': # no set d-separate A and B

                # Either direct causation or latent common cause must exist
                cnf.append([
                    var_mapping[(n1, n2, 'direct')],
                    var_mapping[(n2, n1, 'direct')],
                    var_mapping[(n1, n2, 'latent')]
                ])

            elif edge_type == '<->': # There is a latent common cause of A and B

                # Must have latent common cause
                cnf.append([var_mapping[(n1, n2, 'latent')]])

                # No direct causation in either direction
                cnf.append([-var_mapping[(n2, n1, 'direct')]])
                cnf.append([-var_mapping[(n1, n2, 'direct')]])

    return cnf

def add_transitive_closure_constraints(all_nodes, var_mapping):
    cnf = []

    # Create mapping for transitive relationships
    for i, node_i in enumerate(all_nodes):
        for j, node_j in enumerate(all_nodes):
            if i != j:
                # Define: transitive(i,j) iff i is an ancestor of j through any path
                var_mapping[(node_i, node_j, 'transitive')] = get_next_var_id()

                # Direct edge implies transitive relationship
                cnf.append([-var_mapping[(node_i, node_j, 'direct')],
                           var_mapping[(node_i, node_j, 'transitive')]])

                # Build transitive relationships
                for k, node_k in enumerate(all_nodes):
                    if i != k and j != k:
                        # If i→k and k→j transitively, then i→j transitively
                        cnf.append([
                            -var_mapping[(node_i, node_k, 'transitive')],
                            -var_mapping[(node_k, node_j, 'transitive')],
                            var_mapping[(node_i, node_j, 'transitive')]
                        ])

    return cnf

def add_no_ancestor_constraints(edges, var_mapping):
    cnf = []

    for n1, n2, edge_type in edges:
        if edge_type == 'o->':  # B is not an ancestor of A
            # Use the transitive relationship variable to enforce no ancestry
            cnf.append([-var_mapping[(n2, n1, 'transitive')]])

    return cnf