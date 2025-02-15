import numpy as np
from pysat.solvers import Minisat22


class CausalSAT:
    """
    Implementation of cSAT algorithm for causal structure learning from
    overlapping datasets using SAT solvers
    """

    def __init__(self, datasets, alpha=0.05):
        self.datasets = datasets
        self.alpha = alpha
        self.variables = self._get_union_variables()
        self.clauses = []
        self.var_map = {}
        self._current_var = 1

        # Initialize MAG structure
        self.mag = {v: set() for v in self.variables}

    def _get_union_variables(self):
        """Get union of variables from all datasets"""
        return sorted(set().union(*[set(d.columns) for d in self.datasets]))

    def _create_sat_variable(self, relation):
        """Create SAT variable for a causal relation"""
        if relation not in self.var_map:
            self.var_map[relation] = self._current_var
            self._current_var += 1
        return self.var_map[relation]

    def _learn_initial_pags(self):
        """Learn initial PAGs using FCI algorithm for each dataset"""
        from causallearn.search.FCIBased import fci

        self.pags = []
        for data in self.datasets:
            _, pag = fci(data.to_numpy(), data.columns.tolist(), self.alpha)
            self.pags.append(pag)

    def _encode_pag_constraints(self, pag):
        """Convert PAG constraints to SAT clauses"""
        # Get adjacency matrix and edge types
        adj_matrix = pag.graph
        edge_types = pag.edges

        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] == 1:
                    x = pag.nodes[i]
                    y = pag.nodes[j]

                    # Create variables for possible edge types
                    circ_dir = self._create_sat_variable(f"{x}_o->{y}")
                    dir_edge = self._create_sat_variable(f"{x}->{y}")
                    bidir_edge = self._create_sat_variable(f"{x}<->{y}")

                    # Add constraints based on edge markings
                    if edge_types[i, j] == 1:  # Circle endpoint
                        self.clauses.append([-circ_dir, dir_edge])
                        self.clauses.append([-circ_dir, bidir_edge])
                    elif edge_types[i, j] == 2:  # Arrowhead
                        self.clauses.append([-dir_edge])
                        self.clauses.append([bidir_edge])
                    elif edge_types[i, j] == 3:  # Tail
                        self.clauses.append([-bidir_edge])
                        self.clauses.append([dir_edge])

    def _add_acyclicity_constraints(self):
        """Ensure MAG is acyclic using transitive closure"""
        path_vars = {}

        # Create path variables for all pairs
        for i in self.variables:
            for j in self.variables:
                if i != j:
                    path_vars[(i, j)] = self._create_sat_variable(f"path_{i}_{j}")

        # Transitive closure constraints
        for k in self.variables:
            for i in self.variables:
                for j in self.variables:
                    if i != j and j != k and i != k:
                        # Path i->j implies path i->k and path k->j
                        edge_var = self._create_sat_variable(f"{i}->{j}")
                        self.clauses.append([-edge_var, path_vars[(i, j)]])
                        self.clauses.append([-path_vars[(i, k)], -path_vars[(k, j)], path_vars[(i, j)]])

        # No self-loops
        for v in self.variables:
            self.clauses.append([-self._create_sat_variable(f"{v}->{v}")])

    def solve(self):
        """Solve the SAT problem and construct MAG"""
        self._learn_initial_pags()

        # Encode constraints from all PAGs
        for pag in self.pags:
            self._encode_pag_constraints(pag)

        # Add acyclicity constraints
        self._add_acyclicity_constraints()

        # Solve with Minisat
        with Minisat22(bootstrap_with=self.clauses) as solver:
            if solver.solve():
                model = solver.get_model()
                self._construct_mag(model)
                return True
            return False

    def _construct_mag(self, model):
        """Build MAG from SAT solution"""
        true_vars = {abs(v) for v in model if v > 0}

        for rel, var in self.var_map.items():
            if var in true_vars:
                if '->' in rel:
                    x, y = rel.split('->')
                    self.mag[x].add((y, '->'))
                elif '<->' in rel:
                    x, y = rel.split('<->')
                    self.mag[x].add((y, '<->'))
                    self.mag[y].add((x, '<->'))


# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Example datasets with overlapping variables
    dataset1 = pd.DataFrame(np.random.randint(0, 2, (100, 3)), columns=['A', 'B', 'C'])
    dataset2 = pd.DataFrame(np.random.randint(0, 2, (100, 2)), columns=['B', 'D'])

    csat = CausalSAT([dataset1, dataset2])
    if csat.solve():
        print("Learned MAG structure:")
        for node, edges in csat.mag.items():
            print(f"{node}: {edges}")
    else:
        print("No consistent MAG found")
