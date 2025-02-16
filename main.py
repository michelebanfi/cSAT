from causallearn.graph.Edge import Edge
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.Dataset import load_dataset
from pysat.formula import CNF
from pysat.solvers import Glucose3
import networkx as nx
import matplotlib.pyplot as plt

data, labels = load_dataset("boston_housing")

g, edges = fci(data)
pdy = GraphUtils.to_pydot(g)
# pdy.write_png('boston_housing.png')

# create an ENUM for the edge types
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


sat_clauses = []
formatted_edges = []
for edge in edges:
    formatted_edges.append((edge.node1.name, edge.node2.name, get_edge(edge)))

print(formatted_edges)

def get_unique_nodes(edges):
    nodes = set()
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
    return nodes

nodes = get_unique_nodes(formatted_edges)

var_mapping = {}
def create_variable_mapping(nodes):
    for n1 in nodes:
        for n2 in nodes:
            for edge_type in ["direct", "latent"]:
                var_mapping[(n1, n2, edge_type)] = len(var_mapping) + 1
    return var_mapping

var_mapping = create_variable_mapping(nodes)

def add_edge_constraints(edges):
    cnf = []
    for n1, n2, edge_type in edges:
            if edge_type == '-->':
                # Direct causation must be true
                cnf.append([var_mapping[(n1, n2, 'direct')]])
                # No latent common cause
                cnf.append([-var_mapping[(n1, n2, 'latent')]])

            elif edge_type == 'o->':
                # n2 cannot be ancestor of n1
                cnf.append([-var_mapping[(n2, n1, 'direct')]])

            elif edge_type == 'o-o':
                # Either direct causation or latent common cause must exist
                cnf.append([
                    var_mapping[(n1, n2, 'direct')],
                    var_mapping[(n1, n2, 'latent')]
                ])

            elif edge_type == '<->':
                # Must have latent common cause
                cnf.append([var_mapping[(n1, n2, 'latent')]])
                # No direct causation in either direction
                cnf.append([-var_mapping[(n2, n1, 'direct')]])
                cnf.append([-var_mapping[(n1, n2, 'direct')]])

    return cnf

cnf = add_edge_constraints(formatted_edges)

def decode_cnf(cnf, var_mapping):
    # Create reverse mapping
    reverse_mapping = {v: k for k, v in var_mapping.items()}

    decoded_clauses = []
    for clause in cnf:
        decoded_clause = []
        for literal in clause:
            # Get the absolute value of the literal to find the variable
            var_num = abs(literal)
            var_info = reverse_mapping[var_num]
            node1, node2, edge_type = var_info

            # Create readable representation
            if literal > 0:
                decoded_clause.append(f"{node1}-{edge_type}->{node2}")
            else:
                decoded_clause.append(f"NOT({node1}-{edge_type}->{node2})")

        decoded_clauses.append(decoded_clause)

    return decoded_clauses

# Function to pretty print the decoded CNF
def print_decoded_cnf(decoded_clauses):
    print("Interpreted CNF formula:")
    print("Each line represents a clause (OR between terms)")
    print("The entire formula is an AND between all clauses")
    print("\nClauses:")
    for i, clause in enumerate(decoded_clauses, 1):
        terms = " OR ".join(clause)
        print(f"{i}. {terms}")

# Use the functions
decoded = decode_cnf(cnf, var_mapping)
print_decoded_cnf(decoded)


def solve_causal_sat(cnf_clauses, var_mapping):
    # Create a CNF formula object
    formula = CNF()
    for clause in cnf_clauses:
        formula.append(clause)

    # Create solver and add the formula
    solver = Glucose3()
    solver.append_formula(formula)

    # Solve the formula
    is_sat = solver.solve()

    if not is_sat:
        return False, None

    # Get the solution
    model = solver.get_model()

    # Create reverse mapping for interpretation
    reverse_mapping = {v: k for k, v in var_mapping.items()}

    # Interpret the solution
    causal_relationships = []
    for var in model:
        var_num = abs(var)
        if var_num in reverse_mapping:
            node1, node2, edge_type = reverse_mapping[var_num]
            if var > 0:  # If the variable is positive in the solution
                causal_relationships.append({
                    'from': node1,
                    'to': node2,
                    'type': edge_type,
                    'exists': True
                })
            else:  # If the variable is negative in the solution
                causal_relationships.append({
                    'from': node1,
                    'to': node2,
                    'type': edge_type,
                    'exists': False
                })

    # Clean up
    solver.delete()
    return True, causal_relationships


# Example usage:
satisfiable, solution = solve_causal_sat(cnf, var_mapping)

if satisfiable:
    print("Solution found!")
    print("\nCausal relationships:")
    # Group relationships by type for clearer output
    direct_causes = [rel for rel in solution if rel['type'] == 'direct' and rel['exists']]
    latent_causes = [rel for rel in solution if rel['type'] == 'latent' and rel['exists']]

    print("\nDirect causal relationships:")
    for rel in direct_causes:
        print(f"{rel['from']} --> {rel['to']}")

    print("\nLatent common causes:")
    for rel in latent_causes:
        print(f"{rel['from']} <-> {rel['to']}")
else:
    print("No solution exists - the constraints are unsatisfiable")

import networkx as nx
import matplotlib.pyplot as plt

def visualize_causal_solution(solution, labels=None):
    """
    Create a visualization of the causal structure from SAT solution

    Args:
        solution: List of dictionaries containing causal relationships
        labels: Optional dictionary mapping variable indices to meaningful names
    """
    G = nx.DiGraph()

    # Add nodes and edges
    direct_edges = [(rel['from'], rel['to']) for rel in solution
                    if rel['type'] == 'direct' and rel['exists']]
    latent_edges = [(rel['from'], rel['to']) for rel in solution
                    if rel['type'] == 'latent' and rel['exists']]

    # Add all nodes first
    nodes = set([node for edge in direct_edges + latent_edges for node in edge])
    G.add_nodes_from(nodes)

    # Add direct edges
    G.add_edges_from(direct_edges)

    # Create the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=1000, alpha=0.7)

    # Draw direct edges
    nx.draw_networkx_edges(G, pos, edgelist=direct_edges,
                          edge_color='blue', arrows=True)

    # Draw latent edges differently (as dashed lines)
    for start, end in latent_edges:
        plt.plot([pos[start][0], pos[end][0]],
                [pos[start][1], pos[end][1]],
                'r--', alpha=0.5)

    # Add labels
    if labels is None:
        labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels)

    # Add legend
    plt.plot([], [], 'b-', label='Direct Causation')
    plt.plot([], [], 'r--', label='Latent Common Cause')
    plt.legend()

    plt.title('Discovered Causal Structure')
    plt.axis('off')
    return plt

def analyze_causal_structure(solution):
    """
    Analyze the causal structure to find important patterns and metrics
    """
    # Create directed graph from direct relationships
    G = nx.DiGraph()
    direct_edges = [(rel['from'], rel['to']) for rel in solution
                    if rel['type'] == 'direct' and rel['exists']]
    G.add_edges_from(direct_edges)

    analysis = {
        'root_causes': [node for node in G.nodes() if G.in_degree(node) == 0],
        'leaf_effects': [node for node in G.nodes() if G.out_degree(node) == 0],
        'mediators': [node for node in G.nodes()
                     if G.in_degree(node) > 0 and G.out_degree(node) > 0],
        'cycles': list(nx.simple_cycles(G)),
        'longest_path': nx.dag_longest_path(G) if nx.is_directed_acyclic_graph(G) else None,
        'confounders': [(rel['from'], rel['to']) for rel in solution
                       if rel['type'] == 'latent' and rel['exists']]
    }
    return analysis

plt = visualize_causal_solution(solution)
plt.show()

analysis = analyze_causal_structure(solution)
print("Root causes:", analysis['root_causes'])
print("Ultimate effects:", analysis['leaf_effects'])
print("Mediating variables:", analysis['mediators'])
print("Confounders:", analysis['confounders'])


# Boston Housing dataset column names
boston_labels = {
    'X0': 'CRIM',     # Crime rate
    'X1': 'ZN',       # Proportion of residential land zoned
    'X2': 'INDUS',    # Proportion of non-retail business acres
    'X3': 'CHAS',     # Charles River dummy variable
    'X4': 'NOX',      # Nitric oxides concentration
    'X5': 'RM',       # Average number of rooms per dwelling
    'X6': 'AGE',      # Proportion of owner-occupied units built prior to 1940
    'X7': 'DIS',      # Weighted distances to employment centers
    'X8': 'RAD',      # Index of accessibility to radial highways
    'X9': 'TAX',      # Full-value property-tax rate
    'X10': 'PTRATIO', # Pupil-teacher ratio
    'X11': 'B',       # Proportion of blacks
    'X12': 'LSTAT',   # % lower status of the population
    'X13': 'MEDV'     # Median value of owner-occupied homes
}

# Function to replace X indices with actual names in the solution
def replace_variable_names(solution, label_mapping):
    new_solution = []
    for rel in solution:
        new_rel = rel.copy()
        # Replace 'from' and 'to' with actual names
        new_rel['from'] = label_mapping.get(rel['from'], rel['from'])
        new_rel['to'] = label_mapping.get(rel['to'], rel['to'])
        new_solution.append(new_rel)
    return new_solution

# Apply the replacement
labeled_solution = replace_variable_names(solution, boston_labels)

# Now you can use the labeled solution with the visualization
plt = visualize_causal_solution(labeled_solution)
plt.show()

# And with the analysis
analysis = analyze_causal_structure(labeled_solution)
print("\nRoot causes:", analysis['root_causes'])
print("\nUltimate effects:", analysis['leaf_effects'])
print("\nMediating variables:", analysis['mediators'])
print("\nConfounders:", analysis['confounders'])