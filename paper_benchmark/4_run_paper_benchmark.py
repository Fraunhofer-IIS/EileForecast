import logging
import yaml
from hydra.core.global_hydra import GlobalHydra
import hydra
from datetime import datetime, timedelta
from eile_forecast.main import main as main
from benchmark_helpers import copy_sn_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

path_to_configs = "/home/dev/projects/paper/conf/"
path_to_outputs_forecasts = "/home/dev/projects/data/outputs/benchmark/"
path_to_data = "/home/dev/projects/data/"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg):

    train_len_context_len_dict = {168: 24, 336: 168, 720: 168, 2160: 336, 4320: 336, 6480: 720}

    for model in [
        "SeasNaive",
        "SeasAgg",
        "XGboost",
        "LinearRegression",
        # "TimesFM", #TimesFM needs to installed extra with poetry add timesfm[pax]
        "AutoArima",
        "AutoETS",
        "LSTM",
    ]:
        seas_values = [24, 168] if model == "SeasNaive" else [""] 
        for seas in seas_values:
            for dataset in ([
                    "Cosmic1",
                    "Cosmic2",
                    "Cosmic3",
                ]
                + [f"LP16_{x}" for x in range(1, 21)]
                + [f"LP17_{x}" for x in range(1, 31)]
            ):
                for train_len in [168, 336, 720, 2160, 4320, 6480]:   
                    logger.info(
                        f"Forecasting load of dataset {dataset} with model {model}, seasonality {seas}, and train length {train_len}."
                    )

                    with open(path_to_configs + "config.yaml", "r") as file:
                        config = yaml.safe_load(file)

                        # Modify config.yaml
                    for item in config["defaults"]:
                        if isinstance(item, dict) and "datasets" in item:
                            item["datasets"] = dataset
                        if isinstance(item, dict) and "models" in item:
                            item["models"] = model
                    config["params"]["train_len"] = train_len
                    config["params"]["save_forecast"] = True

                    # Write the updated config back to the file
                    with open(path_to_configs + "config.yaml", "w") as file:
                        yaml.dump(config, file)

                    # Modify model.yaml
                    if model == "SeasNaive":
                        with open(path_to_configs + "models/SeasNaive.yaml", "r") as file:
                            config = yaml.safe_load(file)

                        config["seas"] = seas
                        with open(path_to_configs + "models/SeasNaive.yaml", "w") as file:
                            yaml.dump(config, file)

                    elif model == "XGboost":
                        with open(path_to_configs + "models/XGboost.yaml", "r") as file:
                            config = yaml.safe_load(file)

                        config["hyperparameter_tuning"] = True
                        config["path_to_configs"] = path_to_configs
                        with open(path_to_configs + "models/XGboost.yaml", "w") as file:
                            yaml.dump(config, file)

                    if (model == "XGboost") | (model == "LinearRegression") | (model == "LSTM"):
                        with open(path_to_configs + f"models/{model}.yaml", "r") as file:
                            config = yaml.safe_load(file)
                        config["past_horizon"] = train_len_context_len_dict[train_len]
                        with open(path_to_configs + f"models/{model}.yaml", "w") as file:
                            yaml.dump(config, file)

                    # Modify dataset.yaml
                    with open(
                        path_to_configs + "datasets/" + dataset + ".yaml",
                        "r",
                    ) as file:
                        config = yaml.safe_load(file)
                    orig_start_date = config["benchmark_start_date"]
                    config["benchmark_start_date"] = datetime.strptime(
                        config["benchmark_start_date"], "%Y-%m-%d %H:%M"
                    ) + timedelta(hours=(6480 - train_len))
                    with open(
                        path_to_configs + "datasets/" + dataset + ".yaml",
                        "w",
                    ) as file:
                        yaml.dump(config, file)

                    print(
                        "Updated configs for dataset {} and model {} and train length {}".format(
                            dataset, model, train_len
                        )
                    )

                    # SeasNaive forecasts are the same for all train length
                    # Therefore we just copy the forecasts of train length 168 starting from the respective date and calculate the aggregate metrics
                    if (model == "SeasNaive") & (train_len > 168):
                        copy_sn_metrics(path_to_outputs_forecasts, dataset, seas, path_to_configs, train_len)
                        continue

                    GlobalHydra.instance().clear()

                    @hydra.main(version_base=None, config_path="../conf", config_name="config")
                    def call_main(cfg):
                        main(cfg)

                    call_main()  # pylint: disable=no-value-for-parameter

                    with open(
                        path_to_configs + "datasets/" + dataset + ".yaml",
                        "r",
                    ) as file:
                        config = yaml.safe_load(file)

                    config["benchmark_start_date"] = orig_start_date
                    with open(
                        path_to_configs + "datasets/" + dataset + ".yaml",
                        "w",
                    ) as file:
                        yaml.dump(config, file)

    print("Done.")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
