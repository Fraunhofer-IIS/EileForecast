from typing import Dict, Optional
from abc import ABC, abstractmethod
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.common import FileDataset
from gluonts.model.forecast import SampleForecast


from eile_forecast.forecast.evaluate_forecast import (
    eval_series,
    single_forecast_to_df,
)


class BaseModel(ABC):
    """
    Base class for univariate forecast models to perform predictions.
    Univariate means that the model does not apply the GluonTS MultivariateGrouper on the dataset
    """

    @abstractmethod
    def __init__(
        self, name: str, forecast_horizon: int, forecast_step: int, past_horizon: Optional[int] = None
    ) -> None:
        self.name = name
        self.forecast_horizon = forecast_horizon
        self.estimator = None
        self.freq = "1h"
        self.forecast_step = forecast_step
        self.past_horizon = past_horizon

    def train(self, dataset: FileDataset) -> None:
        """
        Training is implemented using the pre-implemented method of gluonts estimators.
        If this is not possible, this method has to be overwritten in the model definition.

        Parameters
        ----------
        dataset: FileDataset
           gluonts dataset to train on.
        """
        self.predictor = self.estimator.train(dataset)

    def predict(self, test_pairs_input) -> SampleForecast:
        """
        Predictions are gluonts SampleForecast.

        Args:
            test_pairs_input: past data used as prediction input.
        """
        forecasts = self.predictor.predict(test_pairs_input)
        return forecasts

    def postprocess(self, forecast: SampleForecast) -> SampleForecast:
        """
        Set forecasts smaller than 0 to zero

        Parameters
        ----------
        forecast: SampleForecast
           gluonts sample forecast.
        """
        forecast[0].samples[forecast[0].samples < 0] = 0

        return forecast

    def evaluate(
        self,
        test_data: list[Dict],
        forecast: list[SampleForecast],
    ) -> pd.DataFrame:
        """
        Compute accuracy metrics by comparing actual data to the forecasts along the horizon.

        Parameters
        ----------
        test_data
            list containing true target on the predicted range
        forecast
            list of forecasts on the predicted range

        Returns
        -------
        pd.DataFrame
            DataFrame of metrics
        """
        freq = self.freq
        test_df = [
            pd.DataFrame(
                {
                    "start": pd.date_range(
                        (data["start"]).to_timestamp(),
                        periods=len(data["target"]),
                        freq=freq,
                    ),
                    "target": data["target"],
                }
            )
            for data in test_data
        ]

        [df.set_index("start", inplace=True) for df in test_df]
        test_list = [df.set_index(df.index.to_period(freq=freq)) for df in test_df]

        metrics = eval_series(
            ts_list=test_list,
            fcst_list=forecast,
        )

        self.save_log_metrics(metrics=metrics, test_data_start=str(forecast[0].start_date))
        return metrics

    def save_log_metrics(self, metrics: pd.DataFrame, test_data_start: str) -> None:
        """
        Save metrics to parquet.

        Parameters
        ----------
        metrics: metrics, aggregated over time series (=1) and forecasts (=1)
        test_data_start: start date of the test set for naming
        """
        metrics.columns = metrics.columns.str.replace(r"\[", "_", regex=True).str.replace("]", "", regex=True)

        metrics.to_parquet(f"{self.name}_{test_data_start}_metrics.parquet")

    def plot_prob_forecasts(
        self,
        ts_entry: list[pd.Series],
        forecast_entry: list[SampleForecast],
        labels: list[str] = ["predictions", "observations"],
    ) -> None:
        """
        Plot individual sample forecasts and save them as png.

        Parameters
        ----------
        ts_entry: true time series values
        forecast_entry: forecast time series values
        labels: plot label for forecasts and the truth
        """

        _, observations_label = labels

        if self.past_horizon is not None:
            if self.past_horizon < 10 * self.forecast_horizon:
                plot_length = self.past_horizon + self.forecast_horizon
            else:
                plot_length = 3 * self.forecast_horizon
        else:
            plot_length = 3 * self.forecast_horizon

        _, ax = plt.subplots(1, 1, figsize=(10, 7))
        forecast_entry[0].plot(color="g")
        ts_entry[0][-plot_length:].plot(ax=ax, color="b", label=observations_label)
        plt.grid(which="both")
        plt.legend(loc="upper left")
        plt.title(forecast_entry[0].item_id)
        plotname = f"{self.name}_{str(forecast_entry[0].start_date)}_forecast_plot.png"
        plt.savefig(plotname)

    def forecast_to_df(self, forecast: SampleForecast) -> None:
        """
        Concats forecasts for all time series and saves it in a parquet file.
        Parameters
        ----------
        forecast: gluonts sample forecast
        """
        df_list = []
        for fcst in forecast:
            df = single_forecast_to_df(fcst)
            df_list.append(df)
        df_all_forecasts = pd.concat(df_list)
        df_all_forecasts.fcst_step_date = df_all_forecasts.fcst_step_date.apply(lambda x: x.to_timestamp())
        df_all_forecasts.start_date = df_all_forecasts.start_date.apply(lambda x: x.to_timestamp())

        return df_all_forecasts

    def save_forecast(self, fcst_df: pd.DataFrame) -> None:
        start_date = fcst_df.start_date.iloc[0]
        filename = f"{self.name}_{start_date}_forecast.parquet"
        fcst_df.to_parquet(filename)

    def table_forecast_to_gluonts(self, df: pd.DataFrame) -> SampleForecast:
        """
        Turn forecasts of models that give tabular forecasts into gluonts SampleForecasts.
        Parameters
        ----------
        df: data frame with forecasts
        """

        res = re.split(r"(\D+)", self.freq)
        dates = pd.Series(df.fcst_step_date.unique()).sort_values(ascending=True)[: -(self.forecast_horizon - 1)]
        start_dates = dates[0 :: self.forecast_step]
        df_fcst = pd.DataFrame()
        df_fcst["start_date"] = np.tile(np.repeat(start_dates, self.forecast_horizon), df["item_id"].nunique())
        df_fcst["fcst_step"] = np.tile(
            np.tile(range(1, (self.forecast_horizon + 1)), df_fcst["start_date"].nunique()), df["item_id"].nunique()
        )
        df_fcst["fcst_step_date"] = df_fcst["start_date"] + pd.to_timedelta(
            int(res[0]) * (df_fcst["fcst_step"] - 1), unit=res[1]
        )
        df_fcst["item_id"] = df["item_id"][0]  # we train a local model, hence, we have only one item_id
        df_fcst = df_fcst.merge(df, on=["fcst_step_date", "item_id"], how="left")
        groups = df_fcst.groupby(["item_id", "start_date"])
        forecasts_list = []

        for (item_id, start_date), group_df in groups:
            samples = group_df.drop(["item_id", "start_date", "fcst_step", "fcst_step_date"], axis=1).values
            start_date = pd.Period(start_date, freq=self.freq)
            forecast = SampleForecast(
                info=None, item_id=item_id, start_date=start_date, samples=np.swapaxes(np.array(samples), 0, 1)
            )
            forecasts_list.append(forecast)

        return forecasts_list
