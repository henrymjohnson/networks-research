import numpy as np
import networkx as nx


# the deterioration factor of the benefit with respect to distance in the relationship 
# between agents. The further the distance, the indirect benefit, delta, is raised to 
# the power of the distance between the agents
delta = 0.8


def net_utility(g, i, delta):
    """
    Returns the net utility of player i in network g for delta.
    """
    if not isinstance(g, nx.Graph):
        raise ValueError('g must be a networkx Graph')
    if not isinstance(i, int) or i < 0:
        raise ValueError('i must be a positive integer')
    if not isinstance(delta, float) or delta < 0:
        raise ValueError('delta must be a positive float')

    # get all shortest paths from i to all other nodes in g
    paths = nx.shortest_path(g, i)

    benefits = np.array([delta ** len(paths[j]) for j in paths])
    costs = np.array([g[i][j]['cost'] for j in paths])

    return benefits - costs


def net_social_utility(g, delta):
    """
    Returns the net utilities of all nodes in network g for delta.
    """
    if not isinstance(g, nx.Graph):
        raise ValueError('g must be a networkx Graph')
    if not isinstance(delta, float) or delta < 0:
        raise ValueError('delta must be a positive float')
    
    utilities = np.array([net_utility(g, i, delta) for i in g.nodes()])

    return np.sum(utilities)