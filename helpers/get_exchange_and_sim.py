def get_exchange_and_sim(data_conf):
    """Method for making string containing exchange and sim number
    Example: _NYSE@0"""
    # example of exchange and sim string "_NYSE@0"
    return "_" + data_conf["exchange"] + "@" + str(data_conf["sim_no"])