import pytest
import numpy as np
import networkx as nx

from utility_functions.centrality_measurements import closeness_centrality

class TestClosenessCentrality(object):
    def test_closeness_centrality_input_is_graph(self):
        with pytest.raises(ValueError) as excinfo:
            closeness_centrality('graph')
        assert 'Input must be a networkx graph' in str(excinfo.value)
    def test_closeness_centrality_4_nodes_complete_graph(self):
        graph = nx.complete_graph(4)
        assert pytest.approx(closeness_centrality(graph)) == [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]
    def test_closeness_centrality_4_node_lattice(self):
        lattice_graph = nx.grid_2d_graph(4, 4)
        lattice_centrality = closeness_centrality(lattice_graph)
        assert pytest.approx(lattice_centrality) == [((1, 1), 0.46875), ((1, 2), 0.46875), ((2, 1), 0.46875), ((2, 2), 0.46875), ((0, 1), 0.375), ((0, 2), 0.375), ((1, 0), 0.375), ((1, 3), 0.375), ((2, 0), 0.375), ((2, 3), 0.375), ((3, 1), 0.375), ((3, 2), 0.375), ((0, 0), 0.3125), ((0, 3), 0.3125), ((3, 0), 0.3125), ((3, 3), 0.3125)]
    def test_closeness_centrality_4_node_lattice_rand_nodes_added(self):
        lattice_graph = nx.grid_2d_graph(4, 4)
        for i in range(4):
            np.random.seed(i)
            u = list(lattice_graph.nodes())[np.random.choice(np.arange(0, lattice_graph.number_of_nodes()-1))]
            np.random.seed(i+5)
            v = list(lattice_graph.nodes())[np.random.choice(np.arange(0, lattice_graph.number_of_nodes()-1))]
            lattice_graph.add_edge(u, v)
        lattice_centrality = closeness_centrality(lattice_graph)
        assert pytest.approx(lattice_centrality) == [
            ((2, 2), 0.6),
            ((1, 1), 0.5555555555555556),
            ((0, 3), 0.5172413793103449),
            ((2, 1), 0.5),
            ((1, 2), 0.4838709677419355),
            ((3, 0), 0.45454545454545453),
            ((3, 2), 0.4411764705882353),
            ((0, 1), 0.42857142857142855),
            ((0, 2), 0.42857142857142855),
            ((1, 0), 0.42857142857142855),
            ((1, 3), 0.42857142857142855),
            ((2, 0), 0.42857142857142855),
            ((2, 3), 0.42857142857142855),
            ((3, 1), 0.42857142857142855),
            ((0, 0), 0.3409090909090909),
            ((3, 3), 0.3409090909090909)
            ]