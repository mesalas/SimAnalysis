import sys
from helpers.data_conf import make_data_conf

import scripts.orderbooks as orderbooks
from helpers.get_exchange_and_sim import get_exchange_and_sim

def analysis_kwargs(data_conf: dict) -> dict:
    return {"compression": data_conf["compression"], "skip_footer": data_conf["droplast"]}

def make_midprice(input_path, freq, output_path):
    try:
        order_book = orderbooks.order_book_data(input_path, skip_footer=False, time_zone="America/New_York",
                            open_time="9:30:00", trading_day_length="6:30:00")
    except:
        print("{} not read".format(input_path))
    try:

        order_book.get_resampled_midprice(freq).to_csv(output_path,
                                                                     compression = "gzip",
                                                                       index = False)
    except:
        print("Resampled mid-price data {} not written".format(output_path))

if __name__ == "__main__":
    input_path, freq, output_path = sys.argv[1:] # data_path: "testing/test_data" compression : none or "gzip"
    if input_path.split(".")[-1] == "gz":
        compression = "gzip"
    else:
        compression = None

    make_midprice(input_path, freq, output_path)


