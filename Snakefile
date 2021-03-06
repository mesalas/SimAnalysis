rule all:
  input:
    expand(["../temp_data/tmp/figures/{batch_no}_mp_{freq}.png"], batch_no = [1], freq = "5T")

rule get_analysis_framework:
    output: "scripts/make_midprices.py","scripts/plot_midprice.py"
    shell: "git clone https://github.com/mesalas/SimAnalysis.git scripts"

rule midprice_data:
    input:
         "{path}/{symbol}_NYSE@0_Matching-OrderBook.csv" #,"scripts/BarsFromTrades.py"
    output:
          "{path}/reduced_data/{no}_{symbol}_mp_{freq}.csv.gz"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/make_midprices.py {input} {wildcards.freq} {output}"

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

