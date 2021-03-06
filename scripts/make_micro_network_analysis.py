from scripts.make_network import make_traded_volume_matrix, make_directed_graph
import networkx as nx
import numpy as np
import community
from collections import defaultdict
from scripts.negopy import negopy,nx2df

def calculate_total_node_volume(graph : nx.DiGraph):
    for node in list(graph.nodes):
        graph.nodes[node]["out volume"] = int(sum([graph.edges[e[0], e[1]]["weight"] for e in graph.out_edges(node)]))
        graph.nodes[node]["in volume"] = int(sum([graph.edges[e[0], e[1]]["weight"] for e in graph.in_edges(node)]))
        graph.nodes[node]["volume"] = graph.nodes[node]["out volume"] + graph.nodes[node]["in volume"]
    return graph

def normalize_directional_graph(graph):
    for node in list(graph.nodes):
        for out_edge in list(graph.out_edges(node)):
            graph.edges[out_edge]["volume"] = graph.edges[out_edge]["weight"]
            graph.edges[out_edge]["weight"] = graph.edges[out_edge]["weight"] / graph.nodes[node]["out volume"]


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
    print(total_vol)
    ave_vol = list()
    ave_clique_vol = list()
    ave_clique_density = list()
    ave_pct = list()
    ave_relative_vol_within_clique = list()
    n_commun = 0

    for clique,agents in clique_agents_map.items():
        print("Clique no {}".format(clique))

        clique_graph = graph.subgraph(agents) # make subgraph of clique with volumes

        relative_vol_within_clique = list()
        vol = 0


        for agent in agents:
            vol += np.sum([data["volume"] for data in graph[agent].values()]) #sum volume traded by agent to other agents
            graph.nodes[agent]["community"] = clique #set the community property to the clique number
            relative_vol_within_clique.append(np.sum([data["weight"] for data in clique_graph[agent].values()]))


        pct_of_nodes = 100.*len(agents)/len(graph.nodes())
        print("Pct of Nodes {}".format(pct_of_nodes))
        print("mean relative volume within clique {}".format(np.mean(relative_vol_within_clique)))

        print(vol,vol/total_vol)



        vol_within_clique = np.sum([clique_graph[edge[0]][edge[1]]["volume"] for edge in clique_graph.edges()])
        print(vol_within_clique,vol_within_clique/total_vol)
        density = nx.density(clique_graph)

        print("density {}".format(density))
        if pct_of_nodes >= 5.:
            ave_pct.append(pct_of_nodes)
            ave_vol.append(vol)
            ave_clique_vol.append(vol_within_clique)
            ave_clique_density.append(density)
            ave_relative_vol_within_clique.append(np.mean(relative_vol_within_clique))
            n_commun += 1
    print("number of communities containing more than 5 pct of nodes {}".format(n_commun))
    print("average pct of nodes in community {}".format(np.mean(ave_pct)))
    print("average clique density: {}".format(np.mean(ave_clique_density)))

    print("average volume traded by clique: {}\n fraction of total {}".format(np.mean(ave_vol),np.mean(ave_vol)/total_vol))
    print("average volume traded within clique: {}\n fraction of total {}".format(np.mean(ave_clique_vol),np.mean(ave_clique_vol)/total_vol))
    print("average fraction of traded volume from within clique {}".format(np.mean(ave_relative_vol_within_clique)))
    print("***\n")
    return graph


def make_micro_and_macro_network_analysis(input_path, cutoff,output_path):
    nodes, edges = make_traded_volume_matrix(input_path, cutoff)

    graph = nx.DiGraph()  # Rows are the active agent
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)

    # Calculate the total volume traded by each node
    calculate_total_node_volume(graph)

    make_micro_network_analysis(nodes, edges, graph)
    directed_graph_with_cliques = make_louvian_clustering_analysis(graph)
    #directed_graph_with_cliques = make_negopy_clustering_analysis(graph)
    nx.write_gexf(directed_graph_with_cliques, output_path)








