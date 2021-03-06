def make_data_conf(analysis_no, path_and_simno, data_location):
    return {"exchange" : "NYSE",
             "sim_no": path_and_simno["sim_no"],
             "dir" : data_location + path_and_simno["path"]+ "/",
             "droplast" : True,
             "instruments" : ["ABC", "DEF", "GHI"],
             #"agent to color map" : analysis.make_agent_color_map(),
             "compression" : path_and_simno["compression"],
                                        "reduced_dir" : data_location + "/reduced_data/",
                                        "figures_dir" : data_location + "/figures/",
                                        "analysis_no": analysis_no}