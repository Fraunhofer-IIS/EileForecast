import logging
from typing import Optional
from omegaconf import DictConfig
import hydra
import pandas as pd
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast
from eile_forecast.forecast.split_data import instantiate_dataset
from eile_forecast.forecast.evaluate_forecast import paste_inputs_labels, paste_inputs_labels_xgboost

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def instantiate_model(cfg: DictConfig):
    """ "
    Instantiate the model given by the config with hydra.

    Args:
        cfg (DictConfig): configuration drawn from config.yml and model.yml

    Returns:
        The model of the instantiated class
    """
    model = hydra.utils.instantiate(
        cfg.models,
        forecast_horizon=cfg.params.forecast_horizon,
        forecast_step=cfg.params.forecast_step,
    )
    return model


def train_model(model, training_dataset) -> Predictor:
    """
    Train the model on the training data

    Args:
        model: an instantiated model
        training_dataset: data used for model training (type depends on the model)

    Returns:
        Predictor: the trained model
    """

    logger.info(f"Training the model.")
    model.train(training_dataset)
    predictor = model
    return predictor


def forecast(cfg: DictConfig, model, predictor, test_pairs_input, test_pairs_label):
    """
    Forecast with the trained model (predictor), postprocess the forecast.

    Args:
        cfg (DictConfig): configuration drawn from config.yml and dataset.yml
        model: the model
        predictor: the trained model
        test_pairs_input: model input to create forecasts
        test_pairs_label: ground truth data in the past and during the forecast

    Returns:
        the forecast and ground truth in the right format for evaluation and plotting.
    """
    logger.info(f"Forecasting.")
    forecasts = predictor.predict(test_pairs_input)
    fcst_list = list(forecasts)
    fcst_list = predictor.postprocess(fcst_list)
    fcst_df = model.forecast_to_df(fcst_list)

    if cfg.params.save_forecast:
        model.save_forecast(fcst_df=fcst_df)

    if (
        (predictor.name == "XGboost")
        | (predictor.name == "TimesFM")
        | (predictor.name == "DHRArima")
        | (predictor.name == "LinearRegression")
        | (predictor.name == "LSTM")
    ):
        tst_concat_series, tst_concat_list = paste_inputs_labels_xgboost(test_pairs_input, test_pairs_label)
    else:
        tst_concat_series, tst_concat_list = paste_inputs_labels(test_pairs_input, test_pairs_label)

    return tst_concat_list, tst_concat_series, fcst_list, fcst_df


def evaluate_forecast(model, tst_concat_list, tst_concat_series, fcst_list: list[SampleForecast]) -> pd.DataFrame:
    """
    Evaluate the forecast in fcst_list.

    Args:
        model: forecast model
        tst_concat_list: helping data for evaluation and plotting
        tst_concat_series: helping data for evaluation and plotting
        fcst_list: list containing the forecast

    Returns:
        pd.DataFrame: various metrics of the forecast error
    """
    logger.info("Evaluating forecasts.")
    metrics = model.evaluate(
        test_data=tst_concat_list,
        forecast=fcst_list,
    )

    logger.info("Plotting forecasts.")
    model.plot_prob_forecasts(ts_entry=tst_concat_series, forecast_entry=fcst_list)

    return metrics


def preprocess_data(
    cfg: DictConfig,
    start_date: Optional[str],
    end_date: Optional[str],
    split_date: Optional[str] = None,
    trunc_date: Optional[str] = None,
    start_train_date: Optional[str] = None,
):
    """
    Bring the data into the correct format for the forecasting model:
    Merge the load with external features like holidays and weather,
    aggregate to hourly granuarity, fill missig timestamps.
    Split the data set into training and test set.
    Args:
        cfg (DictConfig): configuration drawn from config.yml and dataset.yml
    Returns:
        training and test data of different types for different models
    """
    dataset = instantiate_dataset(cfg, start_date, end_date, split_date, trunc_date, start_train_date)
    training_dataset, test_pairs_input, test_pairs_label = dataset.get_data_split()
    return training_dataset, test_pairs_input, test_pairs_label


def train_forecast_evaluate(
    cfg: DictConfig,
    training_dataset,
    test_pairs_input,
    test_pairs_label,
    model,
) -> tuple[pd.DataFrame, float]:
    """
    Train the chosen model on the chosen dataset, make forecasts and evaluate them.
    Args:
        cfg (DictConfig): configuration drawn from config.yml and dataset.yml
        training_dataset: data for training the model
        test_pairs_input: model input to create forecasts
        test_pairs_label: ground truth data in the past and during the forecast
    Returns:
        tuple[pd.DataFrame, float]: the forecast and its forecast error
    """
    if model.name == "XGboost":
        if cfg.models.hyperparameter_tuning:
            predictor = train_model(model, training_dataset)
            cfg.models.hyperparameter_tuning = False
            model = instantiate_model(cfg)
            predictor = train_model(model, training_dataset)
        else:
            model = instantiate_model(cfg)
            predictor = train_model(model, training_dataset)
    else:
        predictor = train_model(model, training_dataset)

    tst_concat_list, tst_concat_series, fcst_list, fcst_df = forecast(
        cfg, model, predictor, test_pairs_input, test_pairs_label
    )
    if cfg.params.mode == "benchmark":
        track_metrics = evaluate_forecast(model, tst_concat_list, tst_concat_series, fcst_list)
    else:
        track_metrics = None
    return fcst_df, track_metrics


def eval_benchmark(metrics_list: list) -> None:
    """
    Build the mean of the metrics over the benchmark dataset.

    Args:
        metrics_list (list): list of metrics for each forecast

    Returns:
        None
    """
    agg_metrics = pd.concat(metrics_list)
    agg_metrics.drop(columns=["item_id", "forecast_start"], inplace=True)
    agg_metrics = agg_metrics.mean().to_frame().T
    agg_metrics.to_parquet("agg_metrics.parquet")
    return None
