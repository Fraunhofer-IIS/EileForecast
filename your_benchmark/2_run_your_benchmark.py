import logging
import yaml
from hydra.core.global_hydra import GlobalHydra
import hydra
from eile_forecast.main import main as main


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

path_to_configs = "/home/dev/projects/paper/conf/"  # set the path to your config folder
path_to_outputs_forecasts = "/home/dev/projects/data/outputs/benchmark/"  # path where you want to store benchmark results. A folder with the name of the dataset will be created automatically.


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg):
    dataset_name = "example"  # adjust to your dataset name
    train_len = 27916  # adjust to your train len from the bottom of 1_prepare_your_data_for_benchmark.ipynb

    train_len_context_len_dict = {train_len: 2160}
    # If you do not have a lot of time, we suggest to run "SeasNaive" and "SeasAgg". If there is more time, we suggest "XGBoost"
    # If your training data is longer than 9 months, we suggest also to run "LinearRegression".
    # If your training data is shorter than 3 months, we suggest also to run "TimesFM". Note that you need to install the model first with poetry add timesfm[pax].
    # If you still have time, run also "AutoArima", "AutoETS" and "LSTM".
    for model in ["SeasNaive", "SeasAgg", "XGboost", "LinearRegression", "AutoArima", "AutoETS", "LSTM"]:  # "TimesFM",
        seas_values = [24, 168] if model == "SeasNaive" else [""]
        for seas in seas_values:
            for dataset in [dataset_name]:
                for train_len in [train_len]:
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

                    print(
                        "Updated configs for dataset {} and model {} and train length {}".format(
                            dataset, model, train_len
                        )
                    )
                    GlobalHydra.instance().clear()

                    @hydra.main(version_base=None, config_path="../conf", config_name="config")
                    def call_main(cfg):
                        main(cfg)

                    call_main()  # pylint: disable=no-value-for-parameter

    print("Done.")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
