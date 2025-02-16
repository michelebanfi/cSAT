from typing import List, Tuple, Dict, Set
from pysat.formula import CNF
import networkx as nx


class CausalToSAT:
    def __init__(self, edges: List[Tuple]):
        """
        Initialize converter with edges from FCI algorithm
        edges: list of tuples (node1, node2, edge_type)
        edge_types: '-->', 'o->', 'o-o', '<->'
        """
        self.edges = edges
        self.nodes = self._get_unique_nodes()
        self.var_mapping = {}
        self.cnf = CNF()

    def _get_unique_nodes(self) -> Set:
        """Extract unique nodes from edges"""
        nodes = set()
        for n1, n2, _ in self.edges:
            nodes.add(n1)
            nodes.add(n2)
        return nodes

    def _create_variable_mapping(self):
        """Create mapping of edge possibilities to SAT variables"""
        var_counter = 1
        for n1 in self.nodes:
            for n2 in self.nodes:
                if n1 != n2:
                    # Variable for direct causation n1 -> n2
                    self.var_mapping[(n1, n2, 'direct')] = var_counter
                    var_counter += 1
                    # Variable for latent common cause between n1 and n2
                    self.var_mapping[(n1, n2, 'latent')] = var_counter
                    var_counter += 1

    def add_edge_constraints(self):
        """Add clauses based on observed edge types"""
        for n1, n2, edge_type in self.edges:
            if edge_type == '-->':
                # Direct causation must be true
                self.cnf.append([self.var_mapping[(n1, n2, 'direct')]])
                # No latent common cause
                self.cnf.append([-self.var_mapping[(n1, n2, 'latent')]])

            elif edge_type == 'o->':
                # n2 cannot be ancestor of n1
                self.cnf.append([-self.var_mapping[(n2, n1, 'direct')]])

            elif edge_type == 'o-o':
                # Either direct causation or latent common cause must exist
                self.cnf.append([
                    self.var_mapping[(n1, n2, 'direct')],
                    self.var_mapping[(n1, n2, 'latent')]
                ])

            elif edge_type == '<->':
                # Must have latent common cause
                self.cnf.append([self.var_mapping[(n1, n2, 'latent')]])
                # No direct causation in either direction
                self.cnf.append([-self.var_mapping[(n1, n2, 'direct')]])
                self.cnf.append([-self.var_mapping[(n2, n1, 'direct')]])

    def add_transitivity_constraints(self):
        """Add transitivity constraints for causal relationships"""
        for n1 in self.nodes:
            for n2 in self.nodes:
                for n3 in self.nodes:
                    if n1 != n2 and n2 != n3 and n1 != n3:
                        # If n1->n2 and n2->n3, then n1->n3
                        self.cnf.append([
                            -self.var_mapping[(n1, n2, 'direct')],
                            -self.var_mapping[(n2, n3, 'direct')],
                            self.var_mapping[(n1, n3, 'direct')]
                        ])

    def convert(self) -> CNF:
        """Convert causal graph to SAT problem"""
        self._create_variable_mapping()
        self.add_edge_constraints()
        self.add_transitivity_constraints()
        return self.cnf


def convert_fci_to_sat(g, edges):
    """
    Convert FCI output to SAT problem
    g: Graph object from FCI
    edges: Edge list from FCI
    """
    # Convert edges to format (node1, node2, edge_type)
    formatted_edges = []
    for edge in edges:
        n1, n2 = edge[0], edge[1]
        edge_type = edge[2]
        formatted_edges.append((n1, n2, edge_type))

    # Create and run converter
    converter = CausalToSAT(formatted_edges)
    return converter.convert()