import pandas as pd
import numpy as np
import datetime as dt
import math
from pandas.tseries.offsets import BusinessDay
import logging

class time_processor:
    def __init__(self, start_year,start_month, start_day, start_hour, start_minute, start_sec, start_microsec, open_time ="9:30:00", length_of_trading_day ="6:30:00"):
        self.start_date = dt.datetime(start_year,
                                      start_month,
                                      start_day,
                                      start_hour,
                                      start_minute,
                                      start_sec,
                                      start_microsec)




def date_time_to_sim_time(data_frame: pd.DataFrame, open_time: str = "9:30:00", length_of_trading_day: str = "6:30:00") -> pd.DataFrame:
    """

    :rtype: pd.DataFrame
    """
    business_day_number = data_frame.groupby(pd.Grouper(key = "DateTime",freq = 'B')).ngroup()

    data_frame["simTime"] = business_day_number + (
                data_frame["DateTime"] - data_frame["DateTime"].dt.floor("D") - pd.Timedelta(
            open_time)).values.astype(np.float64) * 1e-9 / pd.Timedelta(length_of_trading_day).seconds
    return data_frame

def sim_time_to_time_delta(sim_time, open_time ="9:30:00", length_of_trading_day ="6:30:00"):
    day_in_ns = (np.floor(sim_time) * pd.Timedelta(length_of_trading_day))
    intraday_frac =  sim_time-np.floor(sim_time)
    intra_day_ns = intraday_frac*pd.Timedelta(length_of_trading_day) #.values.astype(np.int64) #* 1e-9
    return pd.to_timedelta(day_in_ns + intra_day_ns, unit = "ns")


def sim_time_to_date_time(simTime, start_date):
    day_and_frac = math.modf(simTime) # containing the frac of day and trading day day no
    return start_date + BusinessDay(day_and_frac[1]) + pd.Timedelta(day_and_frac[0]*23400*1e9)

def process_timestamps(data_frame,timestamp_col_name,timestamp_units, time_zone, open_time, trading_day_length):
    """
    Method processing simulation timestamps and converting them to python datetime format
    :param data_frame:
    :param timestamp_col_name:
    :param timestamp_units:
    :param time_zone:
    :param open_time:
    :param trading_day_length:
    :return: data frame with "DateTime" column
    """
    data_frame["DateTime"] = pd.to_datetime(data_frame[timestamp_col_name],
                                               utc=True, unit=timestamp_units).dt.tz_convert(
        time_zone)
    return date_time_to_sim_time(data_frame, open_time=open_time, length_of_trading_day=trading_day_length)

def date_time_to_timestamp():
    return None