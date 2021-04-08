import unittest
import sys
sys.path.append("scripts/") #Snake make will runn python scripts from scripts/
import make_final_pl_analysis as make_final_pl_analysis
import plot_daily_pl as plot_daily_pl

#from helpers.data_conf import make_data_conf

class MyTestCase(unittest.TestCase):
    def test_pl_analysis(self):
        dispersed_inputs = [["../temp_data/run94/batch1/working/reduced_data/1_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch1/working/reduced_data/1_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch1/working/reduced_data/1_GHI_agent_daily_volume.csv.gz"],
                  ["../temp_data/run94/batch2/working/reduced_data/2_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch2/working/reduced_data/2_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch2/working/reduced_data/2_GHI_agent_daily_volume.csv.gz"],
                  ["../temp_data/run94/batch3/working/reduced_data/3_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch3/working/reduced_data/3_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch3/working/reduced_data/3_GHI_agent_daily_volume.csv.gz"],
                  ["../temp_data/run94/batch4/working/reduced_data/4_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch4/working/reduced_data/4_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch4/working/reduced_data/4_GHI_agent_daily_volume.csv.gz"]
                  ]

        intermediate_inputs = [["../temp_data/run94/batch5/working/reduced_data/5_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch5/working/reduced_data/5_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch5/working/reduced_data/5_GHI_agent_daily_volume.csv.gz"],
                  ["../temp_data/run94/batch6/working/reduced_data/6_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch6/working/reduced_data/6_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch6/working/reduced_data/6_GHI_agent_daily_volume.csv.gz"],
                  ["../temp_data/run94/batch7/working/reduced_data/7_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch7/working/reduced_data/7_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch7/working/reduced_data/7_GHI_agent_daily_volume.csv.gz"],
                  ["../temp_data/run94/batch8/working/reduced_data/8_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch8/working/reduced_data/8_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch8/working/reduced_data/8_GHI_agent_daily_volume.csv.gz"]
                  ]
        concentrated_inputs = [["../temp_data/run94/batch10/working/reduced_data/10_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch10/working/reduced_data/10_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch10/working/reduced_data/10_GHI_agent_daily_volume.csv.gz"],
                  ["../temp_data/run94/batch11/working/reduced_data/11_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch11/working/reduced_data/11_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch11/working/reduced_data/11_GHI_agent_daily_volume.csv.gz"],
                  ["../temp_data/run94/batch12/working/reduced_data/12_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch12/working/reduced_data/12_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch12/working/reduced_data/12_GHI_agent_daily_volume.csv.gz"],
                  ["../temp_data/run94/batch9/working/reduced_data/9_ABC_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch9/working/reduced_data/9_DEF_agent_daily_volume.csv.gz",
                   "../temp_data/run94/batch9/working/reduced_data/9_GHI_agent_daily_volume.csv.gz"]
                  ]


        output = "../temp_data/run94/final_volume.png"

        make_final_pl_analysis.make_final_pl_analysis(dispersed_inputs,intermediate_inputs,concentrated_inputs, output)

if __name__ == '__main__':
    unittest.main()
