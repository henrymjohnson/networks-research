import numpy as np
import networkx as nx

def closeness_centrality(graph, sort='desc'):
    """
    This function takes a graph and returns the list of closeness centrality values for each node.
    This function is based on the following formula from section 2.2 of Social and Economic Networks
    by Matthew O. Jackson, 2008:
        $frac{n-1}{\sum_{j \ne i} l(i, j)}, \text{where} l(i, j) \text{ is shortest path length between i and j}$

    This function does not take into account directionality of edges. It treats the graph as undirected.
    """
    # check if graph is a networkx graph
    if not isinstance(graph, nx.Graph):
        raise ValueError('Input must be a networkx graph')

    close_central = []
    n = graph.number_of_nodes()
    path_lengths = []
    path_length_sums = []
    for i in np.arange(n):
        for j in np.arange(n):
            if nx.has_path(graph, list(graph.nodes)[i], list(graph.nodes)[j]) == False and i != j:
                path_lengths.append(0)
            elif i != j:
                path_lengths.append(nx.shortest_path_length(graph, source=list(graph.nodes)[i], target=list(graph.nodes)[j]))
        path_length_sums.append(np.sum(path_lengths))
        path_lengths = []
    
    close_central = [0 if path_length_sums[i] == 0 else (n-1)/path_length_sums[i] for i in np.arange(n)]
    close_central_list = [(list(graph.nodes)[i], close_central[i]) for i in np.arange(n)]

    if sort != 'desc':
        return close_central_list

    return sorted(close_central_list, key=lambda x: x[1], reverse=True)
