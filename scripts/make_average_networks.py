import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
from make_micro_network_analysis import calculate_total_node_volume,make_micro_network_analysis,make_louvian_clustering_analysis
def make_average_networks(graph_output_path, summary_output_path,inputs):
    networks = list()
    for input in inputs:
        networks.append(pd.read_csv(input, index_col=0))
    adj_matrix = pd.concat(networks).groupby(level=0).mean()

    #nodes, edges = make_traded_volume_matrix(trades.MatchedOrdersData(input_path), cutoff)

    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph)  # Rows are the active agent
    #graph.add_nodes_from(nodes)
    #graph.add_weighted_edges_from(edges)

    # Calculate the total volume traded by each node
    calculate_total_node_volume(graph)

    #make_micro_network_analysis(nodes, edges, graph)
    directed_graph_with_cliques,clique_results = make_louvian_clustering_analysis(graph)
    clique_results.to_csv(summary_output_path)
    nx.write_gexf(directed_graph_with_cliques, graph_output_path)

    return directed_graph_with_cliques