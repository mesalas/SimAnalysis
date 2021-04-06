import pandas as pd
from scripts import timemethods
import numpy as np
#from tqdm import tqdm
import logging


def get_volume_to_side_by_agent(matched_orders, agent_name, side,sample_freq = "1T"):
    vol_to_side = matched_orders[(matched_orders["active_agent"].str.contains(agent_name)) & (matched_orders[" active_side"] == side)][
        ["DateTime", " passive_fillQty"]].groupby(pd.Grouper(key="DateTime",freq = 'B')).resample(sample_freq,on = "DateTime",).sum().reset_index(level=0,
                                                                                                          drop=True)
    vol_to_side["DateTime"] = vol_to_side.index
    return timemethods.date_time_to_sim_time(vol_to_side)

class MatchedOrdersData:
    def __init__(self, path, compression=None, skip_footer=True, time_zone="America/New_York",
                            open_time="9:30:00", trading_day_length="6:30:00"):
        self.path = path
        self.compression = compression
        self.skip_footer = skip_footer
        self.time_zone = time_zone
        self.open_time = open_time
        self.trading_day_length = trading_day_length

        self.active_agent_col ="active_agent"
        self.passive_agent_col = "passive_agent"

        self.passive_side_col = " passive_side"
        self.active_side_col = " active_side"

        self.active_fill_price_col = " active_fillPrice" # note theres a leading space
        self.passive_fill_qty_col = " passive_fillQty"

        self.matched_orders = self.read_matched_orders()


    def make_matched_order_csv_pos_diff(self, time_index="simTime"):
        ''' Create dataframe of the form:
        " {Timestamps} | AgentName | PosDiff "
        From one that is like this:
        " {Timestamps} | active_agent | passive_agent | passive_fill_qty | passive_side "

        Works in these steps:
            1. First ignore the columns "passive_fill_qty | passive_side " and stack the df such that the columns "active_agent | passive_agent"
        become "AgentType | AgentName".
            2. Create list of "passive_fill_qty" and multiply by (-1) if passive_side == SELL.
            3. Make column "PosDiff" in the df that has values of the list when the AgentType == "passive_agent";  the negative of these values if it 
        is "active_agent".
            4. Drop the column "AgentType".
        '''
        print('Creating matched order dataframe...')

        # Create positions for the difference in position 
        timestamps = ['DateTime', 'Nanos', 'simTime']
        keys = timestamps + [self.passive_agent_col, self.active_agent_col]
        diff_df = self.matched_orders[keys].copy()
        diff_df = diff_df.set_index(timestamps)
        diff_df.columns.name = "AgentType"
        diff_df = diff_df.stack().rename_axis(timestamps + ['AgentType']).rename("AgentName").reset_index()
        logging.debug('Copy the dataframe with pertinent columns:')
        
        # Debugging
        debugging_name_map = [True] * len(diff_df)
        logging.debug(diff_df[debugging_name_map])

        passive_pos_diff_list = list(self.matched_orders[self.passive_side_col].apply(lambda x: -1 if x == "SELL" else 1) * self.matched_orders[self.passive_fill_qty_col])
        diff_df.loc[diff_df['AgentType']==self.passive_agent_col, "PosDiff"] = passive_pos_diff_list
        diff_df.loc[diff_df['AgentType']==self.active_agent_col, "PosDiff"] = [-n for n in passive_pos_diff_list]
        
        # Debugging
        logging.debug('Create active/passive_pos_diff columns:')
        logging.debug(diff_df[debugging_name_map])

        print("Created matched order dataframe!")

        return diff_df.drop(columns='AgentType').set_index(time_index)


    def read_matched_orders(self):
        load_csv_kw_args = {"filepath_or_buffer": self.path, "encoding": "utf-8"}
        if self.compression != None:
            load_csv_kw_args["compression"] = self.compression
            load_csv_kw_args["filepath_or_buffer"] = load_csv_kw_args["filepath_or_buffer"] + ".gz"
        matched_orders = pd.read_csv(**load_csv_kw_args)

        if self.skip_footer == True:
            matched_orders = matched_orders[:-1]

        timestamp_col_name = "Nanos"
        timestamp_units = "ns"
        return timemethods.process_timestamps(matched_orders, timestamp_col_name, timestamp_units, self.time_zone,
                                              self.open_time, self.trading_day_length)

    def make_trade_bars(self, freq : str) -> pd.DataFrame:
        bar_returns = lambda x : np.log(x.iloc[-1]) - np.log(x.iloc[0])
        bar_range = lambda x : np.max(x) - np.min(x)
        daily_grouped_trade_price = self.matched_orders[["DateTime", self.active_fill_price_col]].groupby(
            pd.Grouper(key="DateTime",
                       freq='B'
                       ), sort=False
        )
        bars = list()
        for name,group in daily_grouped_trade_price:
            group = group.set_index("DateTime", drop=False)
            group = group.resample(freq
                   ).agg({self.active_fill_price_col :["first", "last", np.max, np.min, bar_range, bar_returns],
                          "DateTime" : ["first", "last"]}) #.reset_index(level=0,
            group.columns = ['open', 'close', 'high', 'low', "range", "return", "first", "last"] #rename columns
            #group.columns = group.columns.droplevel(0)
            bars.append(group)#group.rename(columns={"first" : "open",
                                 #             "last" : "close",
                                 #             'amax': 'high',
                                 #             'amin': 'low',
                                 #             "<lambda_0>" : "range",
                                 #             "<lambda_1>" : "return"}) )                           #drop=True))
        bars = pd.concat(bars)
        bars["DateTime"] = bars.index
        return timemethods.date_time_to_sim_time(bars).reset_index(drop = True)
    def select_data_in_windows(self,windows):
        self.matched_orders.index = self.matched_orders["DateTime"]
        selected_trades = list()
        for window in windows:
            try:
                selected_trades.append(self.matched_orders.loc[window[0]:window[1]])
            except:
                continue
        self.matched_orders = pd.concat(selected_trades)

    def get_unique_agent_classes(self):
        return pd.Series([s.split('-')[0] for s in self.matched_orders[self.active_agent_col]]).unique()

    def get_unique_agent_names(self):
        return self.matched_orders[self.active_agent_col].unique()

    def get_active_agent_trade_prices(self, agent_name):
        return self._get_data_by_active_agent_name( agent_name, self.active_fill_price_col)

    def get_passive_agent_trade_prices(self, agent_name):
        return self._get_data_by_passive_agent_name( agent_name, self.active_fill_price_col)

    def get_active_agent_data(self,agent_name):
        return self._get_data_by_active_or_passive(self.active_agent_col,agent_name)

    def get_passive_agent_data(self,agent_name):
        return self._get_data_by_active_or_passive(self.passive_agent_col,agent_name)

    def get_relative_trade_price_range(self, timewindow):
        grouped_trade_price = self.matched_orders[["DateTime", self.active_fill_price_col]].groupby(
            pd.Grouper(key="DateTime", freq=timewindow))
        grouped_trade_price_max = grouped_trade_price.max()
        grouped_trade_price_min = grouped_trade_price.min()

        rel_range = (grouped_trade_price_max - grouped_trade_price_min) / grouped_trade_price_min
        rel_range["DateTime"] = rel_range.index
        rel_range.reset_index(drop=True, inplace = True)
        rel_range.rename(columns={self.active_fill_price_col: "rel_range"}, inplace = True)
        return timemethods.date_time_to_sim_time(rel_range)
    def get_daily_close(self):
        grouped_trade_price_last = self.matched_orders[["DateTime", self.active_fill_price_col]].groupby(
            pd.Grouper(key="DateTime",
                       freq='B'
                       ), sort=False
        ).last()
        grouped_trade_price_last["DateTime"] = grouped_trade_price_last.index
        grouped_trade_price_last.reset_index(drop=True, inplace = True)
        grouped_trade_price_last.rename(columns={self.active_fill_price_col: "price"}, inplace = True)
        return timemethods.date_time_to_sim_time(grouped_trade_price_last)

    def get_last_traded_price(self, timewindow):
        grouped_trade_price_last = self.matched_orders[["DateTime", self.active_fill_price_col]].groupby(
            pd.Grouper(key="DateTime",
                   freq='B'
                   ), sort = False
        ).resample(timewindow,
        on = "DateTime"
        ).last().reset_index(level=0,
        drop = True)

        grouped_trade_price_last["DateTime"] = grouped_trade_price_last.index
        grouped_trade_price_last.reset_index(drop=True, inplace = True)
        grouped_trade_price_last.rename(columns={self.active_fill_price_col: "price"}, inplace = True)
        return timemethods.date_time_to_sim_time(grouped_trade_price_last)

    def _get_resampled_data_by_active_agent_name(self, agent_name, col_name,sample_freq):
        unsampled_data = self.get_active_agent_data(agent_name)
        unsampled_data = unsampled_data[["DateTime", col_name]] #select the col we are interested in
        unsampled_data = unsampled_data.groupby(["DateTime"]).agg({"DateTime" : "last",col_name:"sum"}) # sum any duplicates

        resampled_data = unsampled_data.groupby(pd.Grouper(key="DateTime",
                            freq='B'
                            ), sort=False
                 ).resample(sample_freq ,on  = "DateTime",).sum().reset_index(level = 0,
                                                                                      drop= True)
        resampled_data["DateTime"] = resampled_data.index
        resampled_data = resampled_data.rename(columns={col_name: agent_name})
        return timemethods.date_time_to_sim_time(resampled_data)

    def get_active_agent_resampled_trade_volume(self, agent, freq):
        return self._get_resampled_data_by_active_agent_name(agent, self.passive_fill_qty_col,freq)

    def _get_data_by_active_agent_name(self, agent_name, col_name):
        trades_data_by_agent = self.get_active_agent_data(agent_name)  # Select rows where agent name is in active agent
        return trades_data_by_agent[["simTime", col_name]]

    def _get_data_by_passive_agent_name(self, agent_name, col_name):
        trades_data_by_agent = self.get_passive_agent_data(agent_name)  # Select rows where agent name is in active agent
        return trades_data_by_agent[["simTime", col_name]]

    def _get_data_by_active_or_passive(self,active_passive,agent_name):
        if active_passive == self.active_agent_col:
            return self.matched_orders[self.matched_orders[self.active_agent_col].str.contains(agent_name)]
        if active_passive == self.passive_agent_col:
            return self.matched_orders[self.matched_orders[self.passive_agent_col].str.contains(agent_name)]

    def get_active_order_size(self, agent_name):
        agent_rows = self.get_active_agent_data(agent_name)
        return agent_rows.groupby(" active_id").last()[["DateTime"," active_orderQty"]]

    def get_daily_active_order_size(self,agent_name):
        order_sizes = self.get_active_order_size(agent_name)
        mean = order_sizes.groupby(pd.Grouper(key="DateTime",freq='B'), sort=False).mean()
        stdev = order_sizes.groupby(pd.Grouper(key="DateTime", freq='B'), sort=False).std()
        return mean, stdev

    def make_agent_pair_volumes(self):

        # https://stackoverflow.com/questions/60383372/aggregate-symmetric-pairs-if-they-exist-in-pandas
        sorter = np.sort(self.matched_orders[["active_agent", "passive_agent"]], axis=1)
        return self.matched_orders.groupby([sorter[:,0], sorter[:,1]])[" passive_fillQty"].sum().reset_index().rename(columns = {"level_0":'first_agent', "level_1":'second_agent', " passive_fillQty": "volume"})

    def make_directional_agent_pair_volumes(self):
        return self.matched_orders.groupby(["active_agent","passive_agent"])[" passive_fillQty"].sum().reset_index().rename(columns = {" passive_fillQty": "volume"})






