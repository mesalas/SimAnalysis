import pandas as pd
import timemethods


class order_book_data:

    def __init__(self, path, compression=None, skip_footer=False, time_zone="America/New_York",
                            open_time="9:30:00", trading_day_length="6:30:00"):
        self.path = path
        self.compression = compression
        self.skip_footer = skip_footer
        self.time_zone = time_zone
        self.open_time = open_time
        self.trading_day_length = trading_day_length
        self.close_time = "16:00:00" #TODO: could be found from the length of trading day

        self.active_agent_col ="active_agent"
        self.passive_agent_col = "passive_agent"

        self.best_ask_name = "Ask1"
        self.best_bid_name = "Bid1"

        self.active_fill_price_col = " active_fillPrice" # note theres a leading space
        self.passive_fill_qty_col = " passive_fillQty"

        self.order_book = self.read_order_book_data()

    def read_order_book_data(self,  type = "AMMengineSim"):
        """ Method for reading order book data into an order book data object
        :returns Pandas DataFrame"""
        if type == "AMMengineSim":
            load_csv_kw_args = {"filepath_or_buffer" : self.path, "encoding" : "utf-8"}
            if self.compression is not None:
                load_csv_kw_args["compression"] = self.compression
                load_csv_kw_args["filepath_or_buffer"] = load_csv_kw_args["filepath_or_buffer"] + ".gz"

            order_book_df = pd.read_csv(**load_csv_kw_args)

            if self.skip_footer == True:
                order_book_df = order_book_df[:-1]

            # For AMMEngine order books the UTC timestamp is labeled "Nanos" and is in nano seconds
            timestamp_col_name = "Nanos"
            timestamp_units = "ns"
        else:
            print("Unknown order book type: {}",type)
            return
        # Convert UTC timestamps to datetime with timezone info

        return timemethods.process_timestamps(order_book_df, timestamp_col_name, timestamp_units, self.time_zone, self.open_time, self.trading_day_length)

    def get_midprice(self) -> pd.DataFrame:
        return self._get_midprice(self.order_book,self.best_ask_name, self.best_bid_name)

    def _get_midprice(self,input_order_book : pd.DataFrame ,best_ask_name: str,best_bid_name: str) -> pd.DataFrame:
        """ Calculates the midprice form as the mean of the best ask and best bid. Will return Nan if best bid or ask is missing
        :returns Pandas DataFrame"""
        return pd.DataFrame({"simTime" : input_order_book["simTime"],
                             "DateTime": input_order_book["DateTime"],
                             "midprice" : 0.5*(input_order_book[best_ask_name]+input_order_book[best_bid_name])
                             },
                            index = input_order_book.index)

    def get_resampled_midprice(self,freq:str) -> pd.DataFrame:
        return self._get_midprice(self.resample_ordebook(freq), self.best_ask_name, self.best_bid_name)

    def get_quote(order_book_df, quote_name): #TODO: Why is this here, are we using this?
        return pd.DataFrame({"simTime" : order_book_df["simTime"], quote_name : order_book_df[quote_name]}, index = order_book_df.index)


    def get_resampled_quote_size(self,quote_name, freq = "1T"):
        order_book_df = self.resample_ordebook(freq)
        return pd.DataFrame({"simTime" : order_book_df["simTime"], quote_name+"Qty" : order_book_df[quote_name+"Qty"]}, index = order_book_df.index)
    def get_spread(self):
        return pd.DataFrame({"simTime" : self.order_book["simTime"], "spread" : self.order_book[self.best_ask_name]-self.order_book[self.best_bid_name]}, index = self.order_book.index)

    def _make_day_string_fromDTindex(DTindex):
            day_string = "{}-{:02d}-{:02d}".format(DTindex.year,
                                         DTindex.month,
                                         DTindex.day)
            return day_string

    def resample_ordebook(self, sample_freq : str) -> pd.DataFrame:
            '''Method for re-sampling the order book.

            1: Remove any rows with duplicate DateTime indices and keeps the last
            2: group order book by business day
            3: re-sample each group using pandas resample method. The resampling happens of the datetime column, timestamps
                are shifted 1 sample freq and missing values are forward filled.
            :returns Pandas DataFrame
            '''

            # Remove duplicate timestamps and keep last
            order_book_df = self.order_book[~self.order_book.index.duplicated(keep='last')]

            # Whats going on here?
            #   * First we split the data into groups containing each business day.
            #   * Then resample each group by frequency
            #   * We then take the last values in each resampled group. then we forward fill and finally reset the index because we
            #     get a multiindex by applying the resample on the groupby object (first index is the groupby groups second is
            #     the resample groups)
            order_book_df = order_book_df.groupby(pd.Grouper(key = "DateTime",
                                                             freq = 'B'
                                                             ), sort = False
                                                  ).resample(sample_freq,
                                                             on  = "DateTime",
                                                             loffset = pd.Timedelta(sample_freq)
                                                             ).last().ffill().reset_index(level = 0,
                                                                                          drop= True)
            order_book_df["DateTime"] = order_book_df.index
            order_book_df =  order_book_df.between_time(self.open_time,self.close_time)

            return timemethods.date_time_to_sim_time(order_book_df)
