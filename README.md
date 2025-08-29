Welcome to the EileForecast package! The package provides a forecasting benchmark framework to compare the accuracy of the most frequently offered load forecasting methods within open-source energy management systems and short-term load forecasting packages. 
This repository makes the benchmark of the paper (link follows after publication) fully reproducible, where we benchmarked the mentioned methods on public SME data. To run the benchmark on your own data, we have prepared examplatory notebooks on public data in the folder your_benchmark.

# Folders
* In /conf the general config as well as dataset and model specific config is set. The config files will be produced and set when running the files in /paper_benchmark and /your_benchmark. You do not need to set any config manually. 
* In /paper_benchmark you can reproduce the benchmark from our paper. Just run the notebooks and scripts sequentially. Remember to set the paths in the files beforehand.
* In /example you can run a benchmark on exemplatory load data from kaggle. You need to transfer the example to your dataset to find out which model works best on your data. Just run the notebooks and scripts sequentially. Remember to set the paths in the files beforehand.
* In /src/eile_forecast lies the source code of EileForecast. 
* In /data you can put the data you want to run the benchmark on. See requirements for the structure.

# Requirements
To run any benchmark, you need the package manager poetry (https://python-poetry.org/) installed. 
Then, build a poetry environment from the pyproject.toml with 'poetry build', 'poetry install' in the terminal.
If you want to run the foundation model TimesFM, additionally install "poetry add timesfm[pax]" (timesfm = {extras = ["pax"], version = "1.3.0"} will be added to your pyproject.toml.)
For the paper benchmark you need around 150GB RAM and for the example benchmark around 20GB.

Further, you need a /data folder at the same level as the /paper_benchmark folder. You need the following /data structure:

data/
├── raw/
├── processed/
└── outputs/
    └── benchmark/






