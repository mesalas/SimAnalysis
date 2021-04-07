import unittest
import sys
sys.path.append("scripts/") #Snake make will runn python scripts from scripts/
import make_average_networks as make_average_networks
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




class MyTestCase(unittest.TestCase):
    def test_pl_analysis(self):
        hm_fig,hm_ax = plt.subplots(nrows=2, ncols=3, figsize = (20,10))
        fig,ax = plt.subplots(nrows=2, ncols=3, figsize = (20,10))
        dispersed_inputs = [pd.read_csv("../temp_data/run93/batch{}/working/reduced_data/{}_ABC_directed_graph_sum.csv".format(num,num)) for num in range(1,5)]

        pd.concat(dispersed_inputs).groupby(level=0).mean().to_csv("../temp_data/run93/dispersed_full_net_stats.csv")

        intermediate_inputs = [pd.read_csv("../temp_data/run93/batch{}/working/reduced_data/{}_ABC_directed_graph_sum.csv".format(num,num)) for num in range(5,9)]
        pd.concat(intermediate_inputs).groupby(level=0).mean().to_csv("../temp_data/run93/intermediate_full_net_stats.csv")
        concentrated_inputs = [pd.read_csv("../temp_data/run93/batch{}/working/reduced_data/{}_ABC_directed_graph_sum.csv".format(num,num)) for num in range(9,13)]
        pd.concat(concentrated_inputs).groupby(level=0).mean().to_csv("../temp_data/run93/concentrated_full_net_stats.csv")

if __name__ == '__main__':
    unittest.main()