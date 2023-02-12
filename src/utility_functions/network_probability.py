import numpy as np
import scipy as sp


def probability_of_network(number_of_nodes, degree, probability):
    """
    Returns the probability of a network being formed given the probability of an edge between two nodes.
    """
    if not (isinstance(number_of_nodes, int) and number_of_nodes > 0):
        raise ValueError('number_of_nodes must be a positive integer')
    if not isinstance(degree, int) or degree < 0:
        raise ValueError('degree must be a positive integer')
    if probability < 0 or probability > 1:
        raise ValueError('probability must be a value between 0 and 1')
    if degree > number_of_nodes * (number_of_nodes - 1) / 2:
        raise ValueError('degree must be less than the maximum possible number of edges, {}.'.format(number_of_nodes * (number_of_nodes - 1) / 2))

    return np.power(probability, degree) * np.power(1 - probability, number_of_nodes * (number_of_nodes - 1) / 2 - degree)


def degree_distribution_of_random_network(number_of_nodes, degree, probability):
    """
    Returns the degree distribution of a random network.
    """
    if not (isinstance(number_of_nodes, int) and number_of_nodes > 0):
        raise ValueError('number_of_nodes must be a positive integer')
    if not isinstance(degree, int) or degree < 0:
        raise ValueError('degree must be a positive integer')
    if probability < 0 or probability > 1:
        raise ValueError('probability must be a value between 0 and 1')

    return sp.special.binom(number_of_nodes - 1, degree) * np.power(probability, degree) * np.power(1 - probability, number_of_nodes - 1 - degree)


def degree_distribution_approximation(number_of_nodes, degree, probability):
    """
    For large enough networks with small link probabilities, the degree distribution of a random network can be approximated 
    by a Poisson distribution. So the fraction of nodes with links numbering the degree can be approximated by the following
    """
    if not (isinstance(number_of_nodes, int) and number_of_nodes > 0):
        raise ValueError('number_of_nodes must be a positive integer')
    if not isinstance(degree, int) or degree < 0:
        raise ValueError('degree must be a positive integer')
    if probability < 0 or probability > 1:
        raise ValueError('probability must be a value between 0 and 1')

    return np.exp(-(number_of_nodes - 1) * probability) * ((number_of_nodes - 1) * probability) ** degree / sp.special.factorial(degree)
