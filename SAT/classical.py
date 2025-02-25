from pysat.formula import CNF
from pysat.solvers import Glucose3

def solveClassicalSAT(cnf):
    formula = CNF(from_clauses=cnf)
    
    solver = Glucose3()
    
    solver.append_formula(formula)
    
    is_sat = solver.solve()
    
    model = solver.get_model()
    
    solver.delete()
    
    return is_sat, model