def validate(cnf: list[list], solution: list) -> bool:
    """
    Validate if a solution satisfies a CNF formula
    
    Args:
        cnf: CNF formula as a list of clauses (each clause is a list of literals)
        solution: List of literals (positive for True, negative for False)
        
    Returns:
        bool: True if the solution satisfies the CNF formula, False otherwise
    """
    # Convert solution to a dictionary for O(1) lookup
    solution_dict = {}
    for lit in solution:
        solution_dict[abs(lit)] = lit > 0
    
    # Check each clause
    for clause in cnf:
        # A clause is satisfied if at least one of its literals is True
        clause_satisfied = False
        
        for lit in clause:
            var = abs(lit)
            # If the literal is positive, it's true when the variable is true
            # If the literal is negative, it's true when the variable is false
            is_positive = lit > 0
            
            # Check if this literal makes the clause true
            if var in solution_dict and (solution_dict[var] == is_positive):
                clause_satisfied = True
                break
        
        # If no literal in this clause is true, the whole formula is unsatisfied
        if not clause_satisfied:
            return False
    
    # All clauses are satisfied
    return True


def validate_all_solutions(cnf: list[list], solutions: list[list]) -> list[bool]:
    """
    Validate multiple solutions against a CNF formula
    
    Args:
        cnf: CNF formula as a list of clauses
        solutions: List of solutions to validate
        
    Returns:
        list[bool]: List of validation results (True for valid, False for invalid)
    """
    return [validate(cnf, solution) for solution in solutions]

# Validate solutions based on the o-> constraint
def has_path(graph, start, end, visited=None):
    """
    Check if there's a path from start to end in the graph using DFS.
    Returns True if there is a path (meaning end is an ancestor of start).
    """
    if visited is None:
        visited = set()
    
    if start == end:
        return True
    
    visited.add(start)
    
    # If start node has no outgoing edges
    if start not in graph:
        return False
    
    for neighbor in graph[start]:
        if neighbor not in visited and has_path(graph, neighbor, end, visited):
            return True
    
    return False

def is_valid_o_to_solution(solution, reversed_causal_dict, o_to_pairs):
    """
    Check if a solution respects the o-> constraints.
    A o-> B means B is not an ancestor of A.
    """
    # Build the directed graph from the solution
    direct_causes = {}
    for var in solution:
        if var > 0:  # Only consider positive literals
            from_node, to_node, edge_type = reversed_causal_dict[abs(var)]
            if edge_type == "direct":
                if from_node not in direct_causes:
                    direct_causes[from_node] = []
                direct_causes[from_node].append(to_node)
    
    # For each o-> pair, check that there's no path from to_node to from_node
    for from_node, to_node in o_to_pairs:
        if has_path(direct_causes, to_node, from_node):
            return False
    
    return True