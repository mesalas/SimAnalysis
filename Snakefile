rule all: #Get the framework we need-
  input:
    "scripts/make_midprices.py","scripts/plot_midprice.py",expand(["figures/{no}_mp_{freq}.png"], no = 1, freq = "5T")

rule get_analysis_framework:
    output: "scripts/make_midprices.py","scripts/plot_midprice.py"
    shell: "git clone https://github.com/mesalas/SimAnalysis.git scripts"

rule midprice_data:
    input:
         "../data/temp/batch{no}/working/{symbol}_NYSE@0_Matching-OrderBook.csv" #,"scripts/BarsFromTrades.py"
    output:
          "reduced_data/{no}_{symbol}_mp_{freq}.csv.gz"
    conda:
        "envs/deps.yaml"
    shell:
         "python scripts/make_midprices.py {input} {wildcards.freq} {output}"

rule midprice_plots:
    input:
        script = "scripts/make_midprices.py",
        files = "reduced_data/{no}_ABC_mp_{freq}.csv.gz reduced_data/{no}_DEF_mp_{freq}.csv.gz reduced_data/{no}_GHI_mp_{freq}.csv.gz"
    output:
        "figures/{no}_mp_{freq}.png"
    conda:
        "envs/deps.yaml"
    shell:
         "python {input.script} {output} {input.files}"

