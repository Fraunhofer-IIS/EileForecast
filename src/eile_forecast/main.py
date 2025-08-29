import logging
import hydra
import pandas as pd
from eile_forecast.forecast import forecast_load


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    """
    This function benchmarks load forecast models.
    Benchmark a model on a dataset from its start to its end date specified in the dataset config.

    Args:
        cfg: configuration gathered by hydra from the model, dataset and general config files.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if (cfg.datasets.forecast_signal == "load") & (cfg.params.mode == "benchmark"):
        model = forecast_load.instantiate_model(cfg)
        metrics_list = []
        start_date = cfg.datasets.benchmark_start_date
        end_date = cfg.datasets.benchmark_end_date

        for i in pd.date_range(
            start_date,
            str(
                pd.Timestamp(end_date)
                - pd.Timedelta("1h") * cfg.params.train_len
                - pd.Timedelta("1h") * cfg.params.forecast_horizon
            ),
        ):

            split_date = str(i + pd.Timedelta("1h") * cfg.params.train_len)
            start_train_date = str(i)
            trunc_date = str(
                i + pd.Timedelta("1h") * cfg.params.train_len + pd.Timedelta("1h") * cfg.params.forecast_horizon
            )
            training_dataset, test_pairs_input, test_pairs_label = forecast_load.preprocess_data(
                cfg=cfg,
                start_date=start_date,
                end_date=end_date,
                split_date=split_date,
                trunc_date=trunc_date,
                start_train_date=start_train_date,
            )
            _, metrics = forecast_load.train_forecast_evaluate(
                cfg, training_dataset, test_pairs_input, test_pairs_label, model
            )
            metrics_list.append(metrics)

        if cfg.models.name == "XGboost":
            if cfg.models.hyperparameter_tuning:
                return None
        forecast_load.eval_benchmark(metrics_list)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
