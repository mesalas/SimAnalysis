import pandas as pd
import timemethods
import numpy as np


#import trades
#import analysis

#def agents_path(data_conf, symbol):
#    return data_conf["dir"] + symbol + analysis.get_exchange_and_sim(data_conf) + "_Matching-agents.csv"

class agent_data:
    def __init__(self, path, compression=None, skip_footer=False, time_zone="America/New_York",
                            open_time="9:30:00", trading_day_length="6:30:00"):
        self.path = path
        self.compression = compression
        self.skip_footer = skip_footer
        self.time_zone = time_zone
        self.open_time = open_time
        self.trading_day_length = trading_day_length
        self.close_time = "16:00:00" #TODO: could be found from the length of trading day

        self.agents_log = self.read_agents_logs()



    def read_agents_logs(self):
        load_csv_kw_args = {"filepath_or_buffer" : self.path, "encoding" : "utf-8"}
        if self.compression != None:
            load_csv_kw_args["compression"] =self.compression
            load_csv_kw_args["filepath_or_buffer"] = load_csv_kw_args["filepath_or_buffer"] + ".gz"
        agents_logs = pd.read_csv(**load_csv_kw_args)

        if self.skip_footer == True:
            agents_logs = agents_logs[:-1]

        timestamp_col_name = "Nanos"
        timestamp_units = "ns"
        return timemethods.process_timestamps(agents_logs, timestamp_col_name, timestamp_units, self.time_zone, self.open_time, self.trading_day_length)

    def get_agent_names(self,  name_prefix):
        agentnames = list(self.agents_log.active_agent.unique())  # Get unique names
        return [s for s in agentnames if name_prefix in s]

def get_full_agent_info(path, agent_class_names, info ,  filter_names = None):
    agents_results = {}
    for agent_class_name in agent_class_names:
        agents_results[agent_class_name] = []


    agents_log = agent_data(path, skip_footer=True)

    # Filter out unwanted agents
    if filter_names is not None:
        for agent in filter_names:
            agent_name_map = agents_log.agents_log["AgentName"].str.contains(agent)
            agents_log.agents_log = agents_log.agents_log[~agent_name_map]

    for agent_name in agent_class_names:
        agent_name_map = agents_log.agents_log["AgentName"].str.contains(agent_name)
        agents_results[agent_name].append(agents_log.agents_log[agent_name_map][["AgentName", "DateTime", info]].pivot_table(
            columns="AgentName", values=info,
            index="DateTime").groupby(pd.Grouper(freq="B")).last().ffill())
        agents_log.agents_log = agents_log.agents_log[~agent_name_map]

    #combine results across all instruments
    for key in agents_results:
        agents_results[key] = sum(agents_results[key])
    return agents_results


def get_agent_statistic(agents_log, statistic, agent_name, sum=False, time_index="simTime"):
    agent_categories = get_agent_abbreviations().keys()

    if agent_name in agent_categories:
        name_map = agents_log["AgentName"].str.contains(agent_name)
    else:
        name_map = agents_log["AgentName"] == agent_name

    if sum is True:
        return agents_log[name_map].pivot_table(columns="AgentName", values=statistic, index=time_index).ffill().sum(
            axis=1)
    else:
        return agents_log[name_map].pivot_table(columns="AgentName", values=statistic, index=time_index).ffill()

def plot_active_and_passive_volumes(active_ax, passive_ax, matched_orders, fraction = False):
    plot_active_volume(active_ax, matched_orders, fraction)
    plot_passive_volume(passive_ax, matched_orders, fraction)

def plot_active_volume(ax, matched_orders, fraction = False):
    ax.set_ylabel("Active volume")
    time,active,labels = get_daily_trades(matched_orders, "active_agent", fraction = fraction)
    ax.stackplot(time, active , labels=labels)

def plot_passive_volume(ax, matched_orders, fraction = False):
    ax.set_ylabel("Passive volume")
    time, passive,labels = get_daily_trades(matched_orders,"passive_agent", fraction = fraction)
    ax.stackplot(time, passive, labels=labels)

def get_daily_trades(matched_orders, side, fraction = False):
    daily_active_trades = matched_orders.pivot_table(columns=side, values=" passive_fillQty",
                                                     index="DateTime", aggfunc=np.sum).groupby(
        pd.Grouper(freq="B")).sum()
    totalDailyTrades = daily_active_trades.groupby(pd.Grouper(freq="B")).sum().sum(axis=1)
    active = daily_active_trades.T.groupby([s.split('-')[0] for s in daily_active_trades.T.index.values]).sum(axis=1).T
    time = active.index
    labels = active.columns.values
    if fraction == True:
        active = active.T/totalDailyTrades
    else:
        active = active.T
    return time,active,labels

def get_daily_trades(matched_orders, agent, kind, agg_func):
    return matched_orders[matched_orders[kind] == agent ]["DateTime"," passive_fillQty"].groupby(
            pd.Grouper(freq="B"), by ="DateTime").sum()

    #return matched_orders.pivot_table(columns= kind, values=" passive_fillQty",
    #                                                     index="DateTime", aggfunc=agg_func).groupby(
    #        pd.Grouper(freq="B")).sum() # .resample("30T",
        # ).sum().reset_index(level = 0,
        # drop= True)

#def plot_volume_per_agent_type(insts, ex_and_sim, dir):
#    plot_trades(insts,ex_and_sim,dir, np.sum)

# def plot_trades(insts, ex_and_sim,dir, agg_func):
#     for symbol in insts:
#         matched_orders = trades.read_matched_orders(dir + symbol + ex_and_sim + "_Matching-MatchedOrders.csv")
#         fig, ax = plt.subplots(2)
#         ax[0].set_title(symbol + " active volume")
#         daily_active_trades = get_daily_trades(matched_orders, agg_func, "active_agent")
#
#         active = daily_active_trades.T.groupby([s.split('-')[0] for s in daily_active_trades.T.index.values]).sum(
#             axis=1).T
#         # active["DateTime"] = active.index
#         # timemethods.date_time_to_sim_time(active)
#         # active.index = active["simTime"]
#         ax[0].stackplot(active.index, active.T, labels=active.columns.values)
#         ax[0].legend()
#
#         ax[1].set_title(symbol + " passive volume")
#         daily_passive_trades = get_daily_trades(matched_orders, agg_func, "passive_agent")
#         passive = daily_passive_trades.T.groupby([s.split('-')[0] for s in daily_passive_trades.T.index.values]).sum(
#             axis=1).T
#         ax[1].stackplot(passive.index, passive.T, labels=passive.columns.values)
#         ax[1].legend()

#def plot_trades_per_agent_type(insts, ex_and_sim, dir):
#    plot_trades(insts, ex_and_sim,dir, "count")

def get_agent_abbreviations():

    agent_abbreviations =  { 
        "SectorRotateInstitutionLT": "SRI-LT",
        "SectorRotateInstitutionST": "SRI-ST",
        "LongShortInstitutionLT": "LSI-LT",
        "LongShortInstitutionST": "LSI-ST",
        "BreakoutTrendLT": "BOT-LT",
        "BreakoutTrendST": "BOT-ST",
        "ScalperReversionLT": "SCR-LT",
        "ScalperReversionST": "SCR-ST",
        "HighLowTrendLT": "HLT-LT",
        "HighLowTrendST": "HLT-ST",
        "OpeningRangeTrend": "ORT",
        "RsiReversionLT": "RR-LT",
        "RsiReversionST": "RR-ST",
        "AggressorTrendLT": "AT-LT",
        "AggressorTrendST": "AT-ST",
        "PullbackReversionLT": "PBR-LT",
        "PullbackReversionST": "PBR-ST",
        "DailyReversion": "DR",
        "RT": "RT",
        "MarketMaker": "MM",
        "PairsTrader": "PT"
    }

    return agent_abbreviations

def create_abbreviation_name_list(names_list):
    
    agent_abbreviations = get_agent_abbreviations()

    agent_names = []
    for agent in names_list:
        for key in agent_abbreviations:
            if key in agent:
                agent_names.append(agent_abbreviations[key])
    
    return agent_names