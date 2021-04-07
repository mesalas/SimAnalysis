rule all:
  input:
    #expand(["../temp_data/run93/batch{no}/working/figures/{no}_mp_{freq}.png"], no = [i for i in range(1,13)], freq = "5T"),
    #expand(["../temp_data/run93/batch{no}/working/figures/{no}_daily_pl.png"],  no = [i for i in range(1,13)]),
    #expand(["../temp_data/run93/batch{no}/working/figures/{no}_daily_volume.png"],  no = [i for i in range(1,13)]),
    #expand(["../temp_data/run93/batch{no}/working/figures/{no}_{symbol}_volatility_{freq}.png"],  no = [i for i in range(1,13)], symbol = ["ABC", "DEF", "GHI"], freq = ["1T", "60T"]),
    #expand(["../temp_data/run93/batch{no}/working/figures/{no}_{symbol}_{quantile}q_volume_heatmap_{freq}.png"],
    #       no = [i for i in range(1,13)], symbol = ["ABC", "DEF", "GHI"], freq = ["1T", "60T"], quantile=[1,20] ),
    #expand(["../temp_data/run93/batch{no}/working/reduced_data/{no}_{symbol}_{quantile}q_trading_table_{freq}.csv"],
    #       no = [i for i in range(1,13)], symbol = ["ABC", "DEF", "GHI"], freq = ["1T", "60T"], quantile=[1,20] ),
    #expand(["../temp_data/run93/batch{no}/working/reduced_data/{no}_{symbol}_{quantile}q_volume_heatmap_{freq}.csv"],
    #       no = [i for i in range(1,13)], symbol = ["ABC", "DEF", "GHI"], freq = ["1T", "60T"], quantile=[1,20] ),
    expand(["../temp_data/run93/batch{no}/working/reduced_data/{no}_{symbol}_directed_graph.gexf"], no = [i for i in range(1,13)], symbol = ["ABC", "DEF", "GHI"]),
    expand(["../temp_data/run93/batch{no}/working/reduced_data/{no}_{symbol}_directed_graph_sum.csv"], no = [i for i in range(1,13)], symbol = ["ABC", "DEF", "GHI"])

rule get_analysis_framework:
    output: "scripts/make_midprices.py","scripts/plot_midprice.py"
    shell: "git clone https://github.com/mesalas/SimAnalysis.git scripts"

rule make_volatility_and_volume_analysis:
    input:
         "{path}/{symbol}_NYSE@0_Matching-MatchedOrders.csv",
         "{path}/reduced_data/{no}_{symbol}_bars_{freq}.csv.gz"
    output:
          "{path}/reduced_data/{no}_{symbol}_{quantile}q_trading_table_{freq}.csv",
          "{path}/figures/{no}_{symbol}_{quantile}q_volume_heatmap_{freq}.png",
          "{path}/reduced_data/{no}_{symbol}_{quantile}q_volume_heatmap_{freq}.csv"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/make_volatility_and_volume_analysis.py {input} {output} {wildcards.quantile}"

rule bars_data:
    input:
         "{path}/{symbol}_NYSE@0_Matching-MatchedOrders.csv" #,"scripts/BarsFromTrades.py"
    output:
          "{path}/reduced_data/{no}_{symbol}_bars_{freq}.csv.gz"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/make_trade_bars.py {input} {wildcards.freq} {output}"

rule volatility_plots:
    input:
        #script = "scripts/make_midprices.py",
        "{path}/reduced_data/{no}_{symbol}_bars_{freq}.csv.gz"
    output:
        "{path}/figures/{no}_{symbol}_volatility_{freq}.png"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/plot_trade_bar_vol.py {output} {input}"

rule midprice_data:
    input:
         "{path}/{symbol}_NYSE@0_Matching-OrderBook.csv" #,"scripts/BarsFromTrades.py"
    output:
          "{path}/reduced_data/{no}_{symbol}_mp_{freq}.csv.gz"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/make_midprices.py {input} {wildcards.freq} {output}"

rule daily_pl_data:
    input:
         "{path}/{symbol}_NYSE@0_Matching-agents.csv" #,"scripts/BarsFromTrades.py"
    output:
          "{path}/reduced_data/{no}_{symbol}_agent_daily_pl.csv.gz"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/make_pl_analysis.py {input} {output}"

rule daily_volume_data:
    input:
         "{path}/{symbol}_NYSE@0_Matching-agents.csv" #,"scripts/BarsFromTrades.py"
    output:
          "{path}/reduced_data/{no}_{symbol}_agent_daily_volume.csv.gz"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/make_volume_analysis.py {input} {output}"

rule midprice_plots:
    input:
        #script = "scripts/make_midprices.py",
        "{path}/reduced_data/{no}_ABC_mp_{freq}.csv.gz",
                 "{path}/reduced_data/{no}_DEF_mp_{freq}.csv.gz",
                 "{path}/reduced_data/{no}_GHI_mp_{freq}.csv.gz"
    output:
        "{path}/figures/{no}_mp_{freq}.png"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/plot_midprice.py {output} {input}"

rule pl_plots:
    input:
        #script = "scripts/make_midprices.py",
        "{path}/reduced_data/{no}_ABC_agent_daily_pl.csv.gz",
                 "{path}/reduced_data/{no}_DEF_agent_daily_pl.csv.gz",
                 "{path}/reduced_data/{no}_GHI_agent_daily_pl.csv.gz"
    output:
        "{path}/figures/{no}_daily_pl.png"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/plot_daily_pl.py {output} {input}"

rule volume_plots:
    input:
        #script = "scripts/make_midprices.py",
        "{path}/reduced_data/{no}_ABC_agent_daily_volume.csv.gz",
                 "{path}/reduced_data/{no}_DEF_agent_daily_volume.csv.gz",
                 "{path}/reduced_data/{no}_GHI_agent_daily_volume.csv.gz"
    output:
        "{path}/figures/{no}_daily_volume.png"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/plot_daily_volume.py {output} {input}"

rule network_analysis:
    input:
         "{path}/{symbol}_NYSE@0_Matching-MatchedOrders.csv",
    output:
          "{path}/reduced_data/{no}_{symbol}_directed_graph.gexf",
          "{path}/reduced_data/{no}_{symbol}_directed_graph_sum.csv"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/make_micro_network_analysis.py {input} {output}"