import logging
from typing import Optional
import pandas as pd
import hydra
from omegaconf import DictConfig
import gluonts.dataset.split
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import TestData, TrainingDataset
from eile_forecast.forecast.preprocess_data import EileDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GluontsDatasetEile(EileDataset):
    def __init__(
        self,
        model: str,
        mode: str,
        forecast_horizon: int,
        forecast_step: int,
        main_signal_name: str,
        split_date: Optional[str] = None,
        trunc_date: Optional[str] = None,
        start_train_date: Optional[str] = None,
        past_horizon: Optional[int] = None,
        raw_path: Optional[str] = None,
        processed_path: Optional[str] = None,
        benchmark_start_date: Optional[str] = None,
        benchmark_end_date: Optional[str] = None,
        pv_signal_name: Optional[str] = None,
        flexible_load_signals: Optional[list[str]] = None,
        forecast_signal: str = "load",
        add_seasonalities: bool = True,
        add_weather: bool = False,
        add_holidays_and_weekends: bool = False,
        **kwargs
    ):
        """_summary_

        Args:
            model (str): The name of the forecast model. Depending on the model, the training an test data sets will be created.
            mode (str): "benchmark". Train-test-split is done accordingly.
                    "benchmark" produces a benchmark on past data with split_date as seperator for training and test set, trunc_date as end of the test set and start_train_date as start of the train set.
            forecast_horizon (int): Number of timesteps to forecast. Used for creating test data/windows.
            forecast_step (int): The time gap between two consecutive forecasts on the test set.
            main_signal_name (str): The string of the main signal id (network electricity consumption).
            split_date (Optional[str]): Split date separates the training and test set.
                Needed for benchmark mode only.
            trunc_date (Optional[str]): Truncation date for the test set in benchmark mode.
            start_train_date (Optional[str]): Truncation date for the train set in benchmark mode.
            past_horizon (Optional[int]): Number of timesteps the model gets as input, model specific.
            raw_path (Optional[str]): Path where raw data is saved, only available in benchmark mode.
            processed_path (Optional[str]): Path where to save processed data, only available in benchmark mode.
            flexible_load_signals (Optional[list[str]]): List of strings of signals of flexible loads.
            pv_signal_name (Optional[str]): String of signal name of the pv production (only one pv plant is possible).
            forecast_signal (str): Signal to be forecasted (target time series): load or pv. Defaults to load.
            benchmark_start_date (Optional[str]): First date to consider for the benchmark. Only available in benchmark mode.
            benchmark_end_date (Optional[str]): Last date to consider for the benchmark. Only available in benchmark mode.
            add_seasonalities (bool): If true, seasonality will be added in the form of sine and cosine columns for all higher granularities. Defaults to True.
            add_weather (bool): If true, given weather data can be added as external feature. Defaults to False.
            add_holidays_and_weekends (bool): If true, holidays according to the country and state in the config will be added. Defaults to False.
        """
        super().__init__(
            signal_id="signal_id",
            power="power",
            date="date",
            hierarchies=["signal_id"],
            forecast_signal=forecast_signal,
            forecast_horizon=forecast_horizon,
            benchmark_start_date=benchmark_start_date,
            benchmark_end_date=benchmark_end_date,
            granularity="1h",
            add_seasonalities=add_seasonalities,
            add_weather=add_weather,
            add_holidays_and_weekends=add_holidays_and_weekends,
            main_signal_name=main_signal_name,
            flexible_load_signals=flexible_load_signals,
            pv_signal_name=pv_signal_name,
            raw_path=raw_path,
            processed_path=processed_path,
        )

        self.mode = mode
        self.model = model
        self.forecast_horizon = forecast_horizon
        self.past_horizon = past_horizon
        self.forecast_step = forecast_step
        self.features = [str(x) for x in self.features]

        if mode == "benchmark":
            self.trunc_date = pd.to_datetime(trunc_date, format="%Y-%m-%d %H:%M:%S").tz_localize(tz="UTC")
            self.split_date = pd.to_datetime(split_date, format="%Y-%m-%d %H:%M:%S").tz_localize(tz="UTC")
            self.start_train_date = pd.to_datetime(start_train_date, format="%Y-%m-%d %H:%M:%S").tz_localize(tz="UTC")
        else:
            self.trunc_date = None
            self.split_date = None
            self.start_train_date = None

    def calculate_n_rolling_origins(self, df) -> int:
        """
        Calculates the number of windows to forecast that shifts by the forecast step

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        ----------
        int: number of windows we get out of the test set
        """
        steps_after_split = (df.loc[df[self.signal_id] == self.forecast_signal, self.date] > self.split_date).sum()
        windows = ((steps_after_split - self.forecast_horizon) // self.forecast_step) + 1

        return windows

    def apply_train_test_split(self, ds: PandasDataset, df: pd.DataFrame) -> tuple[TrainingDataset, TestData]:
        """
        Applies train test split on the dataset.

        Parameters
        ----------
        ds: PandasDataset
        df: pd.DataFrame

        Returns
        ----------
        Tuple[gluonts.dataset.split.TrainingDataset, gluonts.dataset.split.TestData]
        """

        if self.split_date:
            windows = self.calculate_n_rolling_origins(df)

        else:
            max_date = df[self.date].max() + pd.offsets.Week(n=1, weekday=0)
            self.split_date = max_date - pd.Timedelta(weeks=(self.forecast_horizon + windows) * self.forecast_step)
            windows = 1

        training_dataset, test_template = gluonts.dataset.split.split(
            ds, date=pd.Period(self.split_date, self.granularity)
        )

        test_pairs = test_template.generate_instances(
            prediction_length=self.forecast_horizon,
            windows=windows,
            distance=self.forecast_step,
        )
        return training_dataset, test_pairs

    def get_data_split(self) -> tuple:
        """
        Get the preprocessed data for one forecast of the benchmark (benchmark mode)
        and split into training and test data, according to the needs of the model.

        Returns
        ----------
        Model-specific training dataset, test pairs input, and test pairs label
        """

        df = self.get_tabular_data()

        if self.mode == "benchmark":
            df = df[df[self.date] >= self.start_train_date]
            df = df[df[self.date] <= self.trunc_date]

        if (self.model == "XGboost") | (self.model == "LinearRegression"):

            for i in range(self.forecast_horizon, self.past_horizon + 1, self.forecast_horizon):
                df["target_lag_" + str(i)] = df[self.power].shift(i)

            x = df.drop([self.power], axis="columns")
            y = df[[self.power, self.date, self.level]]

            val_date = df.date.min() + 0.8 * (self.split_date - df.date.min())

            x_train = x[x.date <= self.split_date]
            x_train.set_index([self.level, self.date], inplace=True)
            y_train = y[y.date <= self.split_date]
            y_train.set_index([self.level, self.date], inplace=True)
            x_val = x[(x.date >= val_date) & (x.date <= self.split_date)]
            x_val.set_index([self.level, self.date], inplace=True)
            y_val = y[(y.date >= val_date) & (y.date <= self.split_date)]
            y_val.set_index([self.level, self.date], inplace=True)
            x_test = x[x.date > self.split_date]
            x_test.set_index([self.level, self.date], inplace=True)
            y_test = y[y.date > self.split_date]
            y_test.set_index([self.level, self.date], inplace=True)
            training_dataset = [x_train, y_train, x_val, y_val]
            test_pairs_input = [x_test, y_test]
            test_pairs_label = [y, self.granularity, self.forecast_horizon, self.forecast_step]

        elif self.model == "LSTM":
            x = df.drop([self.power], axis="columns")
            y = df[[self.power, self.date, self.level]]

            x_train = x[x.date <= self.split_date]
            x_train.set_index([self.level, self.date], inplace=True)
            y_train = y[y.date <= self.split_date]
            y_train.set_index([self.level, self.date], inplace=True)
            x_test = x[x.date > self.split_date]
            x_test.set_index([self.level, self.date], inplace=True)
            y_test = y[y.date > self.split_date]
            y_test.set_index([self.level, self.date], inplace=True)
            training_dataset = [x_train, y_train]
            test_pairs_input = [x_test, y_test, x_train, y_train]
            test_pairs_label = [y, self.granularity, self.forecast_horizon, self.forecast_step]

        elif self.model == "TimesFM":
            x = df.drop([self.power], axis="columns")
            y = df[[self.power, self.date, self.level]]

            x_train = x[x.date <= self.split_date]
            x_train.set_index([self.level, self.date], inplace=True)
            y_train = y[y.date <= self.split_date]
            y_train.set_index([self.level, self.date], inplace=True)
            x_test = x[x.date > self.split_date]
            x_test.set_index([self.level, self.date], inplace=True)
            y_test = y[y.date > self.split_date]
            y_test.set_index([self.level, self.date], inplace=True)

            training_dataset = [x_train, y_train]
            test_pairs_input = [y_train, y_test]
            test_pairs_label = [y, self.granularity, self.forecast_horizon, self.forecast_step]

        elif self.model == "DHRArima":
            df.reset_index(drop=True, inplace=True)
            y = df[[self.power, self.date, self.level]]
            y_train = y[y.date < self.split_date]
            y_test = y[y.date > self.split_date]
            training_dataset = y_train
            test_pairs_input = [y_train, y_test, df[self.level][0]]
            test_pairs_label = [y, self.granularity, self.forecast_horizon, self.forecast_step]

        else:
            logger.info("Converting to Pandas dataset.")
            ds = PandasDataset.from_long_dataframe(
                dataframe=df.copy(),
                target=self.power,
                item_id=self.signal_id,
                timestamp=self.date,
                freq=self.granularity,
            )

            logger.info("Applying train-test-split.")
            training_dataset, test_pairs = self.apply_train_test_split(ds, df)

            test_pairs_input = test_pairs.input
            test_pairs_label = test_pairs.label

        return training_dataset, test_pairs_input, test_pairs_label


def instantiate_dataset(
    cfg: DictConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    split_date: Optional[str] = None,
    trunc_date: Optional[str] = None,
    start_train_date: Optional[str] = None,
) -> GluontsDatasetEile:
    """
    Instantiate the gluonts data set with hydra.
    Args:
        cfg (DictConfig): configuration drawn from config.yml and dataset.yml
    Returns:
        GluontsDatasetEile: project specific gluonts data set
    """
    if not "past_horizon" in cfg.models:
        past_horizon = None
    else:
        past_horizon = cfg.models.past_horizon
    dataset: GluontsDatasetEile = hydra.utils.instantiate(
        cfg.datasets,
        benchmark_start_date=start_date,
        benchmark_end_date=end_date,
        split_date=split_date,
        trunc_date=trunc_date,
        start_train_date=start_train_date,
        model=cfg.models.name,
        forecast_horizon=cfg.params.forecast_horizon,
        past_horizon=past_horizon,
        forecast_step=cfg.params.forecast_step,
        mode=cfg.params.mode,
        raw_path=cfg.params.raw_path,
        processed_path=cfg.params.processed_path,
    )
    dataset.set_config(cfg)
    return dataset
