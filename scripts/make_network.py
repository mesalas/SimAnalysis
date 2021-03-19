import networkx as nx
from scripts.trades import MatchedOrdersData
import sys
import numpy as np

def make_traded_volume_matrix(input_path, cutoff):

    agents_log = MatchedOrdersData(input_path)  # Read matched orders files

    # Make matrix of trades between agents and list of agent names. The number of trades will be used as the strength of the connections
    # The names will be the nodes
    agent_volumes = agents_log.make_directional_agent_pair_volumes()
    groups = agent_volumes.groupby("active_agent")
    total_volume_for_first_agent = groups.transform(np.sum)
    #if normalize == True:
    #    agent_volumes["volume"] = agent_volumes["volume"] / total_volume_for_first_agent["volume"]

    nodes = agent_volumes["active_agent"].append(agent_volumes["passive_agent"]).unique()
    edges = [t for t,total in zip(agent_volumes.itertuples(index=False, name=None),total_volume_for_first_agent["volume"]) if t[2] > cutoff*total and t[0] != t[1]]

    return nodes,edges

def make_directed_graph(nodes,edges, normalize = True):

        #nodes,edges = make_traded_volume_matrix(input_path, cutoff)

        graph = nx.DiGraph() #Rows are the active agent
        graph.add_nodes_from(nodes)
        graph.add_weighted_edges_from(edges)
        if normalize == True:
                adjacency_df = nx.to_pandas_adjacency(graph)
                graph = nx.from_pandas_adjacency(adjacency_df.div(adjacency_df.sum(axis=1), axis=0).fillna(0)) # We can end up dividing with zero so we need to fill that
        return graph

def save_directed_graph(graph, output):
        nx.write_gexf(graph, output)

if __name__ == "__main__":
    input_path, cutoff, output_path = sys.argv[1:]
    traded_volume_matrix = make_traded_volume_matrix(input_path, cutoff)
    directed_graph = make_directed_graph(traded_volume_matrix[0],traded_volume_matrix[1])
    save_directed_graph(directed_graph, output_path)