import unittest
import sys
sys.path.append("scripts/") #Snake make will runn python scripts from scripts/
import make_trade_bars as make_trade_bars
import trades

class MyTestCase(unittest.TestCase):
    def test_trades_make_bars(self):
        inputs = "testing/test_data/ABC_NYSE@0_Matching-MatchedOrders.csv"
        trades_data = trades.MatchedOrdersData(inputs)
        bars = trades_data.make_trade_bars("15T")
        print(bars.head())

    def test_make_bars(self):
        inputs = ["testing/test_data/ABC_NYSE@0_Matching-MatchedOrders.csv",
                  "testing/test_data/DEF_NYSE@0_Matching-MatchedOrders.csv",
                  "testing/test_data/GHI_NYSE@0_Matching-MatchedOrders.csv"]
        outputs = ["testing/test_data/reduced_data/0_ABC_bars_15T.csv.gz",
                   "testing/test_data/reduced_data/0_DEF_bars_15T.csv.gz",
                   "testing/test_data/reduced_data/0_GHI_bars_15T.csv.gz"]

        [make_trade_bars.make_trade_bars(input_file, "15T", output_file) for input_file,output_file in zip(inputs,outputs)]

if __name__ == '__main__':
    unittest.main()
