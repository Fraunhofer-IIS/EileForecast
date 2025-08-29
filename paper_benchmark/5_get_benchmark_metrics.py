import os
import pandas as pd
from datetime import datetime

# Create a dataframe of all the metrics over the datasets, models and training length.

benchmark_output_folder = "/home/dev/projects/data/outputs/benchmark/"
benchmark_results_folder = benchmark_output_folder + "results/"

datasets = ["Cosmic1", "Cosmic2", "Cosmic3"] + [f"LP16_{x}" for x in range(1, 21)] + [f"LP17_{x}" for x in range(1, 31)]
forecast_signal = "load"
models = [
    "SeasNaive24",
    "SeasNaive168",
    "SeasAgg",
    "AutoArima",
    "LinearRegression",
    "XGboost",
    # "TimesFM",
    "AutoETS",
    "LSTM",
]
train_lengths = [168, 336, 720, 2160, 4320, 6480]

single_metrics = []
for model in models:
    print(f"model: {model}")
    for train_len in train_lengths:
        print(f"train len: {train_len}")

        for data in datasets:
            print(f"dataset: {data}")
            ref_directory = f"{benchmark_output_folder}{data}/train_6480/SeasAgg/"

            dates = []
            for filename in os.listdir(ref_directory):
                if filename.endswith("_metrics.parquet") and filename != "agg_metrics.parquet":
                    date_str = filename.split("_")[1]
                    date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
                    dates.append(date)

            min_date = min(dates)  # evaluate all train lengths from the same timestep of the series
            max_date = max(dates)

            directory = f"{benchmark_output_folder}{data}/train_{train_len}/{model}/"

            for filename in os.listdir(directory):
                if filename.endswith("_metrics.parquet") and filename != "agg_metrics.parquet":
                    date_str = filename.split("_")[1]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")

                    if (file_date >= min_date) & (file_date <= max_date):
                        file_path = os.path.join(directory, filename)
                        met = pd.read_parquet(file_path)
                        met["Model"] = model
                        met["Train Length"] = train_len
                        met["Data"] = data
                        single_metrics.append(met)

metrics = pd.concat(single_metrics, ignore_index=True)[["RMSSE", "Model", "Train Length", "Data"]]
metrics.replace(
    {"SeasNaive24": "SNaive24", "SeasNaive168": "SNaive168", "SeasAgg": "SeasAvg"},
    inplace=True,
)
metrics.to_parquet((benchmark_results_folder + "all_results.parquet"))
print("done")
