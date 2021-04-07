
from make_network import make_traded_volume_matrix, make_directed_graph
import networkx as nx
import pandas as pd
import numpy as np
import community
from collections import defaultdict
from negopy import negopy,nx2df
import trades
import sys

def calculate_total_node_volume(graph : nx.DiGraph):
    for node in list(graph.nodes):
        graph.nodes[node]["out_volume"] = int(sum([graph.edges[e[0], e[1]]["weight"] for e in graph.out_edges(node)]))
        graph.nodes[node]["in_volume"] = int(sum([graph.edges[e[0], e[1]]["weight"] for e in graph.in_edges(node)]))
        graph.nodes[node]["volume"] = graph.nodes[node]["out_volume"] + graph.nodes[node]["in_volume"]
    return graph

def normalize_directional_graph(graph):
    for node in list(graph.nodes):
        for out_edge in list(graph.out_edges(node)):
            graph.edges[out_edge]["volume"] = graph.edges[out_edge]["weight"]
            graph.edges[out_edge]["out fraction"] = graph.edges[out_edge]["weight"] / graph.nodes[node]["out_volume"]


    #adjacency_df = nx.to_pandas_adjacency(graph)
    #normalized_adjacency_df = adjacency_df.div(adjacency_df.sum(axis=1), axis=0).fillna(0)
    #return nx.from_pandas_adjacency(normalized_adjacency_df, create_using= nx.DiGraph)

def make_micro_network_analysis(nodes, edges,graph):

    print("average number of relationships per node {}".format(sum([i[1] for i in graph.out_degree()])/len(graph.out_degree())))
    print("average volume per relationship {}".format(np.mean([e[2] for e in edges])))
    print("total number of nodes {}".format(len(nodes)))
    print("Total number of links {}".format(len(edges)))

def make_negopy_clusters(graph):
    df = nx2df(graph)
    cliques = negopy(df, graph)
    return cliques

def make_louvian_clustering_analysis(graph):
    return make_macro_network_analysis(graph, community.best_partition)
def make_negopy_clustering_analysis(graph):
    return make_macro_network_analysis(graph, make_negopy_clusters)

def make_macro_network_analysis(graph, clustering_method):
    normalize_directional_graph(graph)
    part = clustering_method(nx.to_undirected(graph))

    print("Number of communities {}".format(np.max([part[i] for i in part.keys()])))
    clique_agents_map = defaultdict(list)


    # Make dict. key is clique number val is list of members
    for key,val in sorted(part.items()):
        clique_agents_map[val].append(key)

    # Volume traded by clique
    total_vol = np.sum([graph[edge[0]][edge[1]]["volume"] for edge in graph.edges()])

    commun_results = list()
    for clique,agents in clique_agents_map.items():
        #print("Clique no {}".format(clique))

        clique_graph = graph.subgraph(agents) # make subgraph of clique with volumes

        relative_vol_within_clique = list()
        vol = 0


        for agent in agents:
            vol += np.sum([data["volume"] for data in graph[agent].values()]) #sum volume traded by agent to other agents
            graph.nodes[agent]["community"] = clique #set the community property to the clique number
            relative_vol_within_clique.append(np.sum([data["out fraction"] for data in clique_graph[agent].values()]))

        pct_of_nodes = 100.*len(agents)/len(graph.nodes())
        vol_within_clique = np.sum([clique_graph[edge[0]][edge[1]]["volume"] for edge in clique_graph.edges()])
        density = nx.density(clique_graph)

        commun_results.append(pd.Series([pct_of_nodes,
                                            np.mean(relative_vol_within_clique),
                                            density, vol, total_vol,vol_within_clique,vol_within_clique/vol]))

    commun_results = pd.DataFrame(commun_results)
    commun_results.columns = ["pct nodes in clique",
                                                          "ave rel volume in clique",
                                                          "density",
                                                          "comun volume",
                                                          "total volume",
                                                          "volume within",
                                                          "rel volume within"]
    commun_results=commun_results.sort_values(by=["pct nodes in clique"], ascending=False)

    #print(commun_results)
    #print("number of communities containing more than 5 pct of nodes {}".format(n_commun))
    #print("average pct of nodes in community {}".format(np.mean(ave_pct)))
    #print("average clique density: {}".format(np.mean(ave_clique_density)))

    #print("average volume traded by clique: {}\n fraction of total {}".format(np.mean(ave_vol),np.mean(ave_vol)/total_vol))
    #print("average volume traded within clique: {}\n fraction of total {}".format(np.mean(ave_clique_vol),np.mean(ave_clique_vol)/total_vol))
    #print("average fraction of traded volume from within clique {}".format(np.mean(ave_relative_vol_within_clique)))
    #print("***\n")
    return graph,commun_results


def make_micro_and_macro_network_analysis(input_path, cutoff,graph_output_path,summary_output_path):
    nodes, edges = make_traded_volume_matrix(trades.MatchedOrdersData(input_path), cutoff)

    graph = nx.DiGraph()  # Rows are the active agent
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)

    # Calculate the total volume traded by each node
    calculate_total_node_volume(graph)

    make_micro_network_analysis(nodes, edges, graph)
    directed_graph_with_cliques,clique_results = make_louvian_clustering_analysis(graph)
    #directed_graph_with_cliques = make_negopy_clustering_analysis(graph)
    clique_results.to_csv(summary_output_path)
    nx.write_gexf(directed_graph_with_cliques, graph_output_path)

if __name__ == "__main__":
    input_path, graph_output_path, summary_output_path = sys.argv[1:]
    make_micro_and_macro_network_analysis(input_path, 0.05, graph_output_path, summary_output_path)








