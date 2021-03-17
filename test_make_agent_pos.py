import unittest
import scripts.make_agent_positions as make_agent_positions
import scripts.plot_agent_positions as plot_agent_positions
from helpers.data_conf import make_data_conf

class MyTestCase(unittest.TestCase):
    def test_agent_pos_and_plot_data(self):
        inputs = ["testing/test_data/ABC_NYSE@0_Matching-Agents.csv"]
        output_prefix = ["testing/test_data/reduced_data/0_ABC_pos"]

        output = [make_agent_positions.make_agent_positions(input_file, output_file) for input_file,output_file in zip(inputs,output_prefix)]

        inputs = output
        output = "testing/test_data/figures/0_ABC_agent_pos.png"

        plot_agent_positions.make_mpl_position_plot(output, inputs[0])

        output = "testing/test_data/figures/0_ABC_agent_pos.html"

        plot_agent_positions.make_mpld3_position_plot(output, inputs[0])


if __name__ == '__main__':
    unittest.main()
