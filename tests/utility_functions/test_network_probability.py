import pytest
from utility_functions.network_probability import probability_of_network

class TestProbabilityOfNetwork(object):
    def test_prob_fifty_two_nodes_one_edge(self):
        assert pytest.approx(probability_of_network(2, 1, 0.5)) == 0.5
    def test_prob_zero_two_nodes_one_edge(self):
        assert pytest.approx(probability_of_network(2, 1, 0)) == 0
    def test_prob_hundred_two_nodes_one_edge(self):
        assert pytest.approx(probability_of_network(2, 1, 1)) == 1
    def test_prob_fifty_two_nodes_two_edges(self):
        assert pytest.approx(probability_of_network(2, 1, 0.5)) == 0.5
    def test_prob_fifty_three_nodes_one_edge(self):
        assert pytest.approx(probability_of_network(3, 1, 0.5)) == 0.125

    def test_prob_fifty_zero_nodes_zero_edges(self):
        with pytest.raises(ValueError) as excinfo:
            probability_of_network(0, 0, 0.5)
        assert 'number_of_nodes must be a positive integer' in str(excinfo.value)
    def test_negative_prob_with_two_nodes_one_edge(self):
        with pytest.raises(ValueError) as excinfo:
            probability_of_network(2, 1, -0.5)
        assert 'link_formation_probability must be a value between 0 and 1' in str(excinfo.value)
    
    def test_prob_fifty_and_two_nodes_with_negative_one_edge(self):
        with pytest.raises(ValueError) as excinfo:
            probability_of_network(2, -1, 0.5)
        assert 'number_of_edges must be a positive integer' in str(excinfo.value)
    def test_edges_more_than_maximum_possible(self):
        with pytest.raises(ValueError) as excinfo:
            probability_of_network(2, 3, 0.5)
        assert 'number_of_edges must be less than the maximum' in str(excinfo.value)