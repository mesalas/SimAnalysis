import pandas as pd
import scripts.agents as agents

def make_agent_positions(agents_log_path,output_prefix):

    try:
        agents_log = agents.agent_data(agents_log_path
                                       ).agents_log
    except:
        print("did not get agent data")
    output_files = list()
    agent_types = pd.Series([s.split('-')[0] for s in agents_log["AgentName"]]).unique()
    for agent in agent_types:
        agent_name = agent.split('-')[0]
        try:
            agent_pos = agents.get_agent_statistic(agents_log, "Pos", agent_name, sum = True)
        except:
            print(agents_log_path,agent_name)
            continue
        output_file = output_prefix + "_" + agent_name + ".csv.gz"
#        agent_pos.reset_index(inplace=True)
        agent_pos.to_csv(output_file, compression = "gzip")
        output_files.append(output_file)
    return output_files
        #color, lw = _get_agent_plot_style(data_conf["agent to color map"],agent)