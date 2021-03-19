import unittest
import scripts.make_network as make_network

class MyTestCase(unittest.TestCase):
    def test_resample_and_plot_data(self):
        inputs = ["testing/test_data/ABC_NYSE@0_Matching-MatchedOrders.csv",
                  "testing/test_data/DEF_NYSE@0_Matching-MatchedOrders.csv",
                  "testing/test_data/GHI_NYSE@0_Matching-MatchedOrders.csv"]

        outputs = ["testing/test_data/reduced_data/0_ABC_graph.00.gexf",
                   "testing/test_data/reduced_data/0_DEF_graph_00.gexf",
                   "testing/test_data/reduced_data/0_GHI_graph_00.gexf"]

        [make_network.save_directed_graph(make_network.make_directed_graph(input_file, 0.0), output_file) for input_file,output_file in zip(inputs,outputs)]

if __name__ == '__main__':
    unittest.main()
