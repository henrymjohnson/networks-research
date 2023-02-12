import pytest
import numpy as np

from utility_functions.network_probability import probability_of_network, degree_distribution_of_random_network, degree_distribution_approximation


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
        assert 'probability must be a value between 0 and 1' in str(excinfo.value)
    
    def test_prob_fifty_and_two_nodes_with_negative_one_edge(self):
        with pytest.raises(ValueError) as excinfo:
            probability_of_network(2, -1, 0.5)
        assert 'degree must be a positive integer' in str(excinfo.value)
    def test_edges_more_than_maximum_possible(self):
        with pytest.raises(ValueError) as excinfo:
            probability_of_network(2, 3, 0.5)
        assert 'degree must be less than the maximum' in str(excinfo.value)


class TestDegreeDistributionOfRandomNetwork(object):
    def test_prob_fifty_one_node_one_edge(self):
        assert pytest.approx(degree_distribution_of_random_network(1, 1, 0.5)) == 0
    def test_prob_fifty_two_nodes_one_edge(self):
        assert pytest.approx(degree_distribution_of_random_network(2, 1, 0.5)) == 0.5
    def test_prob_zero_two_nodes_one_edge(self):
        assert pytest.approx(degree_distribution_of_random_network(2, 1, 0)) == 0
    def test_prob_hundred_two_nodes_one_edge(self):
        assert pytest.approx(degree_distribution_of_random_network(2, 1, 1)) == 1
    def test_prob_fifty_two_nodes_two_edges(self):
        assert pytest.approx(degree_distribution_of_random_network(2, 2, 0.5)) == 0
    def test_prob_fifty_three_nodes_one_edge(self):
        assert pytest.approx(degree_distribution_of_random_network(3, 1, 0.5)) == 0.5

    def test_prob_fifty_zero_nodes_zero_edges(self):
        with pytest.raises(ValueError) as excinfo:
            degree_distribution_of_random_network(0, 0, 0.5)
        assert 'number_of_nodes must be a positive integer' in str(excinfo.value)
    def test_negative_prob_with_two_nodes_one_edge(self):
        with pytest.raises(ValueError) as excinfo:
            degree_distribution_of_random_network(2, 1, -0.5)
        assert 'probability must be a value between 0 and 1' in str(excinfo.value)
    
    def test_prob_fifty_and_two_nodes_with_negative_one_edge(self):
        with pytest.raises(ValueError) as excinfo:
            degree_distribution_of_random_network(2, -1, 0.5)
        assert 'degree must be a positive integer' in str(excinfo.value)


class TestDegreeDistributionApproximation(object):
    def test_prob_fifty_one_node_one_edge(self):
        assert pytest.approx(degree_distribution_approximation(1, 1, 0.5)) == 0
    def test_prob_fifty_two_nodes_one_edge(self):
        assert pytest.approx(degree_distribution_approximation(2, 1, 0.5)) == 0.3032653
    def test_prob_zero_two_nodes_one_edge(self):
        assert pytest.approx(degree_distribution_approximation(2, 1, 0)) == 0
    def test_prob_hundred_two_nodes_one_edge(self):
        assert pytest.approx(degree_distribution_approximation(2, 1, 1)) == np.exp(-1)
    def test_prob_fifty_two_nodes_two_edges(self):
        assert pytest.approx(degree_distribution_approximation(2, 2, 0.5)) == 0.0758163
    def test_prob_fifty_three_nodes_one_edge(self):
        assert pytest.approx(degree_distribution_approximation(3, 1, 0.5)) == 0.3678794
    def test_prob_twenty_five_hundred_nodes_five_edges(self):
        assert pytest.approx(degree_distribution_approximation(500, 5, 0.25)) == 0.0000000
    def test_prob_fifty_hundred_nodes_five_edges(self):
        assert pytest.approx(degree_distribution_approximation(10, 2, 0.75)) == 0.0266741

    def test_prob_fifty_zero_nodes_zero_edges(self):
        with pytest.raises(ValueError) as excinfo:
            degree_distribution_approximation(0, 0, 0.5)
        assert 'number_of_nodes must be a positive integer' in str(excinfo.value)
    def test_negative_prob_with_two_nodes_one_edge(self):
        with pytest.raises(ValueError) as excinfo:
            degree_distribution_approximation(2, 1, -0.5)
        assert 'probability must be a value between 0 and 1' in str(excinfo.value)
    