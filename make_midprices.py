import orderbooks
from helpers.get_exchange_and_sim import get_exchange_and_sim

def analysis_kwargs(data_conf: dict) -> dict:
    return {"compression": data_conf["compression"], "skip_footer": data_conf["droplast"]}

def make_midprice(data_conf, sample_rate = "1T"):
    orderbook_path = lambda data_conf,symbol : data_conf["dir"] + symbol + get_exchange_and_sim(data_conf) + "_Matching-OrderBook.csv"

    # midprice
    for inst in data_conf["instruments"]:
        try:
            order_book = orderbooks.order_book_data(orderbook_path(data_conf, inst), **analysis_kwargs(data_conf))
        except:
            print("{} not read".format(orderbook_path(data_conf, inst)))
        try:
            out_path = data_conf["output_dir"] +str(data_conf["analysis_no"]) + "_" + inst + "_mp_" +  sample_rate+ ".csv.gz"
            order_book.get_resampled_midprice(sample_rate).to_csv(out_path,
                                                                         compression = "gzip",
                                                                           index = False)
        except:
            print("Resampled mid-price data {} not written".format(out_path))


