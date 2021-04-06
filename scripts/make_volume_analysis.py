from agents import get_full_agent_info
import pandas as pd
import sys
import helpers
def make_volume_analysis(input_path,output_path):

    agents = [
        "MarketMaker",
        "RT",
        "SectorRotateInst",
        "LongShortInst",
        "PairsTrader""BreakoutTrend",
                    "HighLowTrend",
                    "OpeningRangeTrend",
                    "AggressorTrend","RsiReversion",
                        "PullbackReversion",
                        "ScalperReversion",
                        "DailyReversion"]


    agents_results = get_full_agent_info(
        input_path,
        info="Vol",
        agent_class_names = agents,
        filter_names = None
    )

    pd.concat([agents_results[key] for key in agents_results], axis = 1).fillna(0.0).to_csv(output_path,
                                                                     compression = "gzip",
                                                                       index = False)
if __name__ == "__main__":
    input_path, output_path = sys.argv[1:]  # data_path: "testing/test_data" compression : none or "gzip"

    make_volume_analysis(input_path, output_path)