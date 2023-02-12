import numpy as np

def probability_of_network(number_of_nodes, number_of_edges, link_formation_probability):
    """
    Returns the probability of a network being formed given the probability of an edge between a set of nodes.
    """
    if not (isinstance(number_of_nodes, int) and number_of_nodes > 0):
        raise ValueError('number_of_nodes must be a positive integer')
    if not isinstance(number_of_edges, int) or number_of_edges < 0:
        raise ValueError('number_of_edges must be a positive integer')
    if link_formation_probability < 0 or link_formation_probability > 1:
        raise ValueError('link_formation_probability must be a value between 0 and 1')
    if number_of_edges > number_of_nodes * (number_of_nodes - 1) / 2:
        raise ValueError('number_of_edges must be less than the maximum possible number of edges, {}.'.format(number_of_nodes * (number_of_nodes - 1) / 2))

    return np.power(link_formation_probability, number_of_edges) * np.power(1 - link_formation_probability, number_of_nodes * (number_of_nodes - 1) / 2 - number_of_edges)
