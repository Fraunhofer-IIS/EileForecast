import os
import shutil
import yaml
import pandas as pd


def copy_sn_metrics(benchmark_folder: str, dataset: str, seas: str, config_path: str, train_len: int):
    source_folder = f"{benchmark_folder}{dataset}/train_168/SeasNaive{seas}/"
    destination_folder = f"{benchmark_folder}{dataset}/train_{train_len}/SeasNaive{seas}/"
    os.makedirs(destination_folder, exist_ok=True)

    with open(
        config_path + "datasets/" + dataset + ".yaml",
        "r",
    ) as file:
        config = yaml.safe_load(file)

    reference_date = pd.Timestamp(config["benchmark_start_date"]) + pd.Timedelta("1h") * train_len + pd.Timedelta("1h")

    for filename in os.listdir(source_folder):
        if filename.endswith("_metrics.parquet") and "agg_metric" not in filename:
            date_str = filename.split("_")[1]
            file_date = pd.Timestamp(date_str)

            if file_date >= reference_date:
                shutil.copy(os.path.join(source_folder, filename), destination_folder)

    calculate_agg_metrics(destination_folder)
    return None


def calculate_agg_metrics(destination_folder: str):
    metrics_lst = []
    for filename in os.listdir(destination_folder):
        if filename.endswith("_metrics.parquet") and filename != "agg_metrics.parquet":
            met = pd.read_parquet(os.path.join(destination_folder, filename))
            metrics_lst.append(met)
    metrics = pd.concat(metrics_lst)
    metrics.drop(columns=["item_id", "forecast_start"], inplace=True)
    agg_metrics = metrics.mean().to_frame().T
    agg_metrics.to_parquet(f"{destination_folder}agg_metrics.parquet")

    return None
