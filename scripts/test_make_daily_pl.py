import unittest
import sys
sys.path.append("scripts/") #Snake make will runn python scripts from scripts/
import make_pl_analysis as make_pl_analysis

#from helpers.data_conf import make_data_conf

class MyTestCase(unittest.TestCase):
    def test_pl_analysis(self):
        inputs = ["testing/test_data/ABC_NYSE@0_Matching-agents.csv",
                  "testing/test_data/DEF_NYSE@0_Matching-agents.csv",
                  "testing/test_data/GHI_NYSE@0_Matching-agents.csv"]
        outputs = ["testing/test_data/reduced_data/0_ABC_agent_daily_pl.csv.gz",
                   "testing/test_data/reduced_data/0_DEF_agent_daily_pl.csv.gz",
                   "testing/test_data/reduced_data/0_GHI_agent_daily_pl.csv.gz"]

        [make_pl_analysis.make_pl_analysis(input_file,output_file) for input_file,output_file in zip(inputs,outputs)]

        #inputs = outputs
        #output = "testing/test_data/figures/0_mp_5T.png"

        #plot_midprice.make_mpl_midprice_plot(output, inputs)

        #output = "testing/test_data/figures/0_mp_5T.html"
        #plot_midprice.make_mpld3_midprice_plot(output, inputs)


if __name__ == '__main__':
    unittest.main()
