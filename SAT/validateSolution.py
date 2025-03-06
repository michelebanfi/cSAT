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