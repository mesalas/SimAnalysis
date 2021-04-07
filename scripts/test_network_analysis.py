import sys
sys.path.append("scripts/")
import unittest
import pandas as pd
import networkx as nx
import scripts.make_micro_network_analysis as make_micro_network_analysis
from scripts.make_micro_network_analysis import normalize_directional_graph,calculate_total_node_volume

class MyTestCase(unittest.TestCase):
    def test_normalize_graph(self):
        matrix = pd.DataFrame([[0,4,2,0,0],[1,0,2,0,0],[1,0,0,0,0], [0,0,1,0,5],[0,0,0,5,0]])
        graph = nx.from_pandas_adjacency(matrix, create_using= nx.DiGraph)
        print(matrix)
        print(nx.to_pandas_adjacency(graph))
        calculate_total_node_volume(graph)
        normalize_directional_graph(graph)
        print(nx.to_pandas_adjacency(graph))



    def test_resample_and_plot_data(self):

        inputs = ["testing/test_data/ABC_NYSE@0_Matching-MatchedOrders.csv"]#,
                  #"testing/test_data/DEF_NYSE@0_Matching-MatchedOrders.csv",
                  #"testing/test_data/GHI_NYSE@0_Matching-MatchedOrders.csv"]
        #inputs = ["../Simulations/run92/batch11/wd/ABC_NYSE@11_Matching-MatchedOrders.csv"]#,
                  #"../Simulations/run87/batch1/working/DEF_NYSE@0_Matching-MatchedOrders.csv",
                  #"../Simulations/run87/batch1/working/GHI_NYSE@0_Matching-MatchedOrders.csv"]
        outputs = ["testing/test_data/reduced_data/92_11_0_ABC_graph_cliques_05.gexf"]#,
                   #"testing/test_data/reduced_data/0_DEF_graph_cliques_05.gexf",
                   #"testing/test_data/reduced_data/0_GHI_graph_cliques_05.gexf"]
        summary_outputs = ["testing/test_data/reduced_data/92_11_0_ABC_graph_cliques_05.csv"]
        [make_micro_network_analysis.make_micro_and_macro_network_analysis(input_file, 0.05,output_file,summary_output) for input_file,output_file,summary_output in zip(inputs,outputs,summary_outputs)]

if __name__ == '__main__':
    unittest.main()