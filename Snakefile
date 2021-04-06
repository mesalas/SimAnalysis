rule all:
  input:
    expand(["../temp_data/run93/batch{no}/working/figures/{no}_mp_{freq}.png"], no = [i for i in range(1,13)], freq = "5T"),
    expand(["../temp_data/run93/batch{no}/working/figures/{no}_daily_pl.png"],  no = [i for i in range(1,13)]),
    expand(["../temp_data/run93/batch{no}/working/figures/{no}_daily_volume.png"],  no = [i for i in range(1,13)])


rule get_analysis_framework:
    output: "scripts/make_midprices.py","scripts/plot_midprice.py"
    shell: "git clone https://github.com/mesalas/SimAnalysis.git scripts"

rule bars_data:
    input:
         "{path}/{symbol}_NYSE@0_Matching-MatchedOrders.csv" #,"scripts/BarsFromTrades.py"
    output:
          "{path}/reduced_data/{no}_{symbol}_bars_{freq}.csv.gz"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/make_trade_bars.py {input} {wildcards.freq} {output}"

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

