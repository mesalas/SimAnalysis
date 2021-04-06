import trades
import sys

def make_trade_bars(input_path: str,freq: str,output: str) -> None:
    """
    function for making bars of trades data
    :param input_path:
    :param freq:
    :param output:
    :return: none
    """

    trades_data = trades.MatchedOrdersData(input_path)
    bars = trades_data.make_trade_bars(freq)
    bars.to_csv(output,index = False)

if "__name__" == "__main__":
    input_path, freq, output_path = sys.argv[1:] # data_path: "testing/test_data" compression : none or "gzip"

    make_trades_bars(input_path, freq, output_path)