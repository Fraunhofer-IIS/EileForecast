import re
import logging
import multiprocessing
import sys
from functools import partial
from typing import Iterable, Optional, Union, Dict, Mapping
import numpy as np
import pandas as pd
from gluonts.evaluation import Evaluator
from gluonts.evaluation._base import validate_forecast, worker_function
from gluonts.model.forecast import SampleForecast
from gluonts.gluonts_tqdm import tqdm
from gluonts.time_feature import get_seasonality
from gluonts.ext.naive_2 import naive_2

from gluonts.evaluation.metrics import (
    calculate_seasonal_error,
    coverage,
    mape,
    mase,
    msis,
    quantile_loss,
    smape,
)


class EvaluatorEile(Evaluator):
    def __call__(
        self,
        ts_iterator: Iterable[Union[pd.DataFrame, pd.Series]],
        fcst_iterator: Iterable[SampleForecast],
        num_series: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compute accuracy metrics by comparing actual data to the forecasts.

        Parameters
        ----------
        ts_iterator
            iterator containing true target on the predicted range
        fcst_iterator
            iterator of forecasts on the predicted range
        num_series
            number of series of the iterator
            (optional, only used for displaying progress)

        Returns
        -------
        dict
            Dictionary of aggregated metrics
        pd.DataFrame
            DataFrame containing metrics per time series
        """
        ts_iterator = iter(ts_iterator)
        fcst_iterator = iter(fcst_iterator)

        rows = []

        with tqdm(
            zip(ts_iterator, fcst_iterator),
            total=num_series,
            desc="Running evaluation",
        ) as it, np.errstate(divide="ignore", invalid="ignore"):
            if self.num_workers and sys.platform != "win32":
                mp_pool = multiprocessing.Pool(initializer=None, processes=self.num_workers)
                rows = mp_pool.map(
                    func=partial(worker_function, self),
                    iterable=iter(it),
                    chunksize=self.chunk_size,
                )
                mp_pool.close()
                mp_pool.join()
            else:
                for ts, forecast in it:
                    rows.append(self.get_metrics_per_ts(ts, forecast))

        assert not any(True for _ in ts_iterator), "ts_iterator has more elements than fcst_iterator"

        assert not any(True for _ in fcst_iterator), "fcst_iterator has more elements than ts_iterator"

        if num_series is not None:
            assert len(rows) == num_series, f"num_series={num_series} did not match number of" f" elements={len(rows)}"

        metrics_per_ts = pd.DataFrame.from_records(rows)

        # If all entries of a target array are NaNs, the resulting metric will
        # have value "masked". Pandas does not handle masked values correctly.
        # Thus we set dtype=np.float64 to convert masked values back to NaNs
        # which are handled correctly by pandas Dataframes during
        # aggregation.
        metrics_per_ts = metrics_per_ts.astype(
            {col: np.float64 for col in metrics_per_ts.columns if col not in ["item_id", "forecast_start"]}
        )

        return metrics_per_ts

    def calculate_seasonal_rmse(
        self,
        past_data: np.ndarray,
        freq: Optional[str] = None,
        seasonality: Optional[int] = None,
    ):
        r"""
        .. math::

        seasonal\_error = mean(|Y[t] - Y[t-m]|)

        where m is the seasonal frequency. See [HA21]_ for more details.
        """
        # Check if the length of the time series is larger than the seasonal
        # frequency
        if not seasonality:
            assert freq is not None, "Either freq or seasonality must be provided"
            seasonality = get_seasonality(freq)

        if seasonality < len(past_data):
            forecast_freq = seasonality
        else:
            # edge case: the seasonal freq is larger than the length of ts
            # revert to freq=1

            # logging.info('The seasonal frequency is larger than the length of the
            # time series. Reverting to freq=1.')
            forecast_freq = 1

        y_t = past_data[:-forecast_freq]
        y_tm = past_data[forecast_freq:]

        return np.sqrt(np.mean(np.square(y_t - y_tm)))  # RMSE of the seasonal naive forecast

    def calculate_seasonal_mae(
        self,
        past_data: np.ndarray,
        freq: Optional[str] = None,
        seasonality: Optional[int] = None,
    ):

        return calculate_seasonal_error(past_data=past_data, freq=freq, seasonality=seasonality)

    def rmse(self, target: np.ndarray, forecast: np.ndarray) -> float:

        return np.sqrt(np.mean(np.square(target - forecast)))

    def mae(self, target: np.ndarray, forecast: np.ndarray) -> float:

        return np.mean(abs(target - forecast))

    def rmsse(
        self,
        rmse: float,
        seasonal_rmse: float,
    ) -> float:
        return rmse / seasonal_rmse

    def get_base_metrics(
        self, forecast: SampleForecast, pred_target, mean_fcst, median_fcst, seasonal_mae, seasonal_rmse
    ) -> Dict[str, Union[float, str, None]]:
        rmse = self.rmse(pred_target, median_fcst)
        mae = self.mae(pred_target, median_fcst)
        return {
            "item_id": forecast.item_id,
            "forecast_start": forecast.start_date,
            "RMSE": rmse,
            "MAE": mae,
            "seasonal_mae": seasonal_mae,
            "seasonal_rmse": seasonal_rmse,
            "MASE": mase(pred_target, median_fcst, seasonal_mae),
            "RMSSE": self.rmsse(rmse, seasonal_rmse),
            "MAPE": mape(pred_target, median_fcst),
            "sMAPE": smape(pred_target, median_fcst),
        }

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: SampleForecast
    ) -> Mapping[str, Union[float, str, None, np.ma.core.MaskedConstant]]:
        if not validate_forecast(forecast, self.quantiles):
            if self.allow_nan_forecast:
                logging.warning("Forecast contains NaN values. Metrics may be incorrect.")
            else:
                raise ValueError("Forecast contains NaN values.")

        pred_target = np.array(self.extract_pred_target(time_series, forecast))
        past_data = np.array(self.extract_past_data(time_series, forecast))

        if self.ignore_invalid_values:
            past_data = np.ma.masked_invalid(past_data)
            pred_target = np.ma.masked_invalid(pred_target)

        try:
            mean_fcst = getattr(forecast, "mean", None)
        except NotImplementedError:
            mean_fcst = None

        median_fcst = forecast.quantile(0.5)

        seasonal_rmse = self.calculate_seasonal_rmse(past_data, forecast.start_date.freqstr, self.seasonality)
        seasonal_mae = self.calculate_seasonal_mae(past_data, forecast.start_date.freqstr, self.seasonality)

        metrics: Dict[str, Union[float, str, None]] = self.get_base_metrics(
            forecast, pred_target, mean_fcst, median_fcst, seasonal_mae, seasonal_rmse
        )

        if self.custom_eval_fn is not None:
            for k, (eval_fn, _, fcst_type) in self.custom_eval_fn.items():
                if fcst_type == "mean":
                    if mean_fcst is not None:
                        target_fcst = mean_fcst
                    else:
                        logging.warning("mean_fcst is None, therefore median_fcst is used.")
                        target_fcst = median_fcst
                else:
                    target_fcst = median_fcst

                try:
                    val = {
                        k: eval_fn(
                            pred_target,
                            target_fcst,
                        )
                    }
                except Exception:
                    logging.warning(f"Error occurred when evaluating {k}.")
                    val = {k: np.nan}

                metrics.update(val)

        try:
            metrics["MSIS"] = msis(
                pred_target,
                forecast.quantile(self.alpha / 2),
                forecast.quantile(1.0 - self.alpha / 2),
                seasonal_mae,
                self.alpha,
            )
        except Exception:
            logging.warning("Could not calculate MSIS metric.")
            metrics["MSIS"] = np.nan

        if self.calculate_owa:
            naive_median_forecast = naive_2(past_data, len(pred_target), freq=forecast.start_date.freqstr)
            metrics["sMAPE_naive2"] = smape(pred_target, naive_median_forecast)
            metrics["MASE_naive2"] = mase(pred_target, naive_median_forecast, seasonal_mae)

        for quantile in self.quantiles:
            forecast_quantile = forecast.quantile(quantile.value)

            metrics[f"QuantileLoss[{quantile}]"] = quantile_loss(pred_target, forecast_quantile, quantile.value)
            metrics[f"Coverage[{quantile}]"] = coverage(pred_target, forecast_quantile)

        return metrics


def bias(target, forecast):
    return np.mean(forecast - target)


def eval_series(
    ts_list: list[pd.DataFrame],
    fcst_list: list[SampleForecast],
) -> pd.DataFrame:
    """
    Compute accuracy metrics of a single series by comparing actual data to the forecasts.

    Parameters
    ----------
    ts_list
        list containing true target on the predicted range
    fcst_list
        list of forecasts on the predicted range

    Returns
    -------
    pd.DataFrame
        DataFrame of metrics over the forecast horizon
    """
    error = {
        "bias": [bias, "mean", "median"],
    }
    univariate_evaluator = EvaluatorEile(custom_eval_fn=error, num_workers=0)

    metrics = univariate_evaluator(iter(ts_list), iter(fcst_list))

    return metrics


def single_forecast_to_df(forecast):
    """
    Creates pandas df from gluonts SampleForecast.
    Columns are item_id, start_date, fcst_step and one column per sample (sample_fcst1, ..., sample_fcstns).
    As a result, the dimensions are (h, 3+ns).
    """
    samples = forecast.samples
    ns, h = samples.shape
    item_id = [forecast.item_id] * h
    start_date = [forecast.start_date] * h
    fcst_step = [i + 1 for i in range(h)]
    fcst_step_date = [forecast.start_date + (step - 1) * forecast.start_date.freq for step in fcst_step]

    df_samples = pd.DataFrame(samples.swapaxes(0, 1))
    df_samples.columns = ["sample_fcst" + str(i) for i in range(ns)]

    df_ids = pd.DataFrame(
        {
            "item_id": item_id,
            "start_date": start_date,
            "fcst_step": fcst_step,
            "fcst_step_date": fcst_step_date,
        }
    )

    df_samples = pd.concat([df_ids, df_samples], axis=1)

    return df_samples


def paste_inputs_labels(inputs, labels) -> tuple[list, list]:
    """
    Concat test_pairs.label to test_pairs.input at the end ('target' values only).
    Then turn the array into a pd.Series object. Make a list of these.
    And pass the list to the plot_prob_forecats.
    """
    tst_concat_series = []
    tst_concat_list = []
    for i, j in zip(inputs, labels):
        start_date = i["start"].to_timestamp()
        freq = i["start"].freq
        concat_val = list(np.concatenate((i["target"], j["target"])))
        index = pd.date_range(start_date, periods=len(concat_val), freq=freq)
        tst_concat_series.append(pd.Series(concat_val, index=index))
        tst_concat_list.append(
            {
                "start": i["start"],
                "target": np.concatenate((i["target"], j["target"])),
                "item_id": i["item_id"],
            }
        )
    return tst_concat_series, tst_concat_list


def paste_inputs_labels_xgboost(inputs, labels) -> tuple[list, list]:
    """
    Concat test_pairs.label to test_pairs.input at the end ('target' values only).
    Then turn the array into a pd.Series object. Make a list of these.
    And pass the list to the plot_prob_forecats.

    inputs = [X_test, y_test]
    labels = [y, freq, forecast_horizon, forecast_step]
    """

    tst_concat_series = []
    tst_concat_list = []
    y = labels[0]
    y_test = inputs[1].reset_index()
    freq = labels[1]
    forecast_horizon = labels[2]
    forecast_step = labels[3]
    res = re.split(r"(\D+)", freq)
    forecast_step_freq = int(res[0]) * forecast_step

    for i in y.signal_id.unique():
        start_date_data = y[y.signal_id == i]["date"].min()
        forecast_start_dates = pd.date_range(
            start=y_test[y_test.signal_id == i]["date"].min(),
            end=y_test[y_test.signal_id == i]["date"].max() - pd.Timedelta((forecast_horizon - 1) * freq),
            freq=(str(forecast_step_freq) + res[1]),
        )

        for start_date in forecast_start_dates:
            forecast_dates = pd.date_range(start=start_date, freq=freq, periods=forecast_horizon)
            concat_val = y[(y.signal_id == i) & (y.date <= forecast_dates[-1])]
            index = pd.date_range(start_date_data, periods=len(concat_val), freq=freq)
            tst_concat_series.append(pd.Series(concat_val["power"].values, index=index))
            tst_concat_list.append(
                {
                    "start": pd.Period(start_date_data, freq=freq),
                    "target": concat_val["power"].values,
                    "item_id": i,
                }
            )
    return tst_concat_series, tst_concat_list
