import os
import logging
from typing import Optional
import pandas as pd
import torch.utils.data
from eile_forecast.helpers.aggregator import Aggregator
from eile_forecast.helpers.datefiller import DateFiller
from eile_forecast.helpers.filter import Filter
from eile_forecast.helpers.merger import Merger


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        signal_id: str,
        power: str,
        date: str,
        forecast_signal: str,
        hierarchies: list[int],
        main_signal_name: int,
        forecast_horizon: int,
        benchmark_start_date: Optional[str],
        benchmark_end_date: Optional[str],
        raw_path: Optional[str],
        processed_path: Optional[str],
        flexible_load_signals: Optional[list[int]] = None,
        pv_signal_name: Optional[list[int]] = None,
        granularity: str = "1h",
        add_seasonalities: bool = True,
        add_weather: bool = True,
        add_holidays_and_weekends: bool = True,
        mode: str = "benchmark",
    ) -> None:

        self.signal_id = signal_id
        self.forecast_signal = forecast_signal
        self.level = self.signal_id
        self.mode = mode
        self.power = power
        self.date = date
        self.forecast_horizon = forecast_horizon
        if (benchmark_start_date != None) & (benchmark_end_date != None):
            self.benchmark_start_date = benchmark_start_date
            self.benchmark_end_date = benchmark_end_date
        self.raw_path = raw_path
        self.processed_path = processed_path

        self.main_signal_name = main_signal_name
        self.flexible_load_signals = flexible_load_signals
        self.pv_signal_name = pv_signal_name

        self.hierarchies = hierarchies
        self.granularity = granularity

        self.add_seasonalities = add_seasonalities
        self.add_weather = add_weather
        self.add_holidays_and_weekends = add_holidays_and_weekends

        self.features = []

        self.aggr_dict = {
            self.power: "mean",
            self.date: "min",
        }
        self.config = None
        self.split_date = None
        self.filter = None
        self.aggregator = None
        self.datefiller = None
        self.merger = None

    def set_config(self, cfg):
        self.config = cfg

    def read_electricity_demand(self) -> pd.DataFrame:
        """
        Read data from the raw path.

        Returns
        ----------
        df: pandas DataFrame with the raw data.
        """
        assert self.raw_path is not None, "Please provide the raw path of the data in dataset.yaml"
        df = pd.read_parquet(self.raw_path + "electricity_demand.parquet")
        df.rename(index={"demand": "load"}, inplace=True)
        return df

    def load_raw_dataset(self) -> pd.DataFrame:
        """
        Read in data, reset index, sort by date.
        Cut to the benchmark dates in the benchmark mode.
        Do not allow negative power.

        Returns
        ----------
        df: pandas DataFrame with the raw data, prepared for further processing.
        """

        df = self.read_electricity_demand()
        df.reset_index(inplace=True)
        df.sort_values(by=["date"], inplace=True)

        if self.mode == "benchmark":
            df = df[(df.date >= self.benchmark_start_date) & (df.date <= self.benchmark_end_date)]

        df.loc[df["power"] < 0, "power"] = 0
        return df

    def get_pv_and_nfl_signal(self, df) -> pd.DataFrame:
        """
        Calculate non-flexible load (nfl) = overall load (main signal) - flexible loads
        Also keep prodcuction signals.
        Rename nfl to load and production to pv.
        Do not allow negative power.
        """
        if self.flexible_load_signals:
            nfl_ids = [[self.main_signal_name], self.flexible_load_signals]
            nfl_signals = [x for xs in nfl_ids for x in xs]
        else:
            nfl_signals = [self.main_signal_name]

        nfl_min_date = max(
            df[df.signal_id.isin(nfl_signals)].groupby("signal_id").date.min(),
        )

        nfl_max_date = min(
            df[df.signal_id.isin(nfl_signals)].groupby("signal_id").date.max(),
        )

        df_main = df[df.signal_id == self.main_signal_name]
        df_main = df_main[(df_main.date >= nfl_min_date) & (df_main.date <= nfl_max_date)]

        if self.flexible_load_signals:
            subtract = []
            for signal_id in self.flexible_load_signals:
                df_fl = df[df.signal_id == signal_id]
                df_fl = df_fl[(df_fl.date >= nfl_min_date) & (df_fl.date <= nfl_max_date)]
                subtract.append(df_fl["power"].values)
        else:
            subtract = None

        df_nfl = df_main.copy()
        df_nfl["power"] = df_main["power"].values
        if subtract:
            for sub in subtract:
                df_nfl["power"] = df_nfl["power"] - sub
        df_nfl["signal_id"] = "load"
        if self.pv_signal_name:
            df_pv = df[df.signal_id == self.pv_signal_name]
            df_pv["signal_id"] = "pv"

            df = pd.concat([df_nfl, df_pv])
        else:
            df = df_nfl
        df.loc[df["power"] < 0, "power"] = 0
        return df

    def get_tabular_data(self) -> pd.DataFrame:
        """
        Load the data from the raw path.
        Preprocess the data:
            Aggregate to hourly data.
            Fill missing time stamps.
            Calculate the non-flexible load, if there is a flexible one.
            Filter for the forecast signal.
            Add weather data.
            Add holidays and weekends.
            Add seasonal variables.
            Save the preprocessed data in the benchmark mode to the processed path.

        Returns
        ----------
        df: pd.DataFrame of the preprocessed data.
        """
        logger.info("Loading and preprocessing data.")
        if (
            (self.mode == "benchmark")
            & (self.processed_path is not None)
            & (os.path.exists(os.path.join(self.processed_path, "processed.parquet")))
        ):
            df = pd.read_parquet(self.processed_path + "processed.parquet")
        else:
            self._init_df_processing_modules()
            df = self.load_raw_dataset()
            df = self.aggregator.aggregate_level_granularity(df)
            df = self.datefiller.get_all_dates(df)
            df = self.get_pv_and_nfl_signal(df)
            df = self.filter.filter(df)
            if self.add_weather:
                df = self.merger.merge_weather(df)
            if self.add_holidays_and_weekends:
                df = self.merger.merge_holidays(df)
            if self.add_seasonalities:
                df, self.features = self.merger.get_periodicity(df)
            df = self.sort_df(df)

            if (self.mode == "benchmark") & (self.processed_path is not None):
                os.makedirs(self.processed_path, exist_ok=True)
                df.to_parquet(self.processed_path + "processed.parquet")
        return df

    def _init_df_processing_modules(self):
        """
        Init the objects necessary for preparing df.
        """
        self.filter = Filter(
            self.forecast_signal,
            self.signal_id,
            self.power,
        )

        self.aggregator = Aggregator(
            self.signal_id,
            self.granularity,
            self.date,
            self.power,
            self.hierarchies,
            aggr_dict=self.aggr_dict,
        )

        self.datefiller = DateFiller(
            self.date, self.granularity, self.signal_id, self.hierarchies, self.power, self.features
        )

        self.merger = Merger(
            date=self.date,
            granularity=self.granularity,
            features=self.features,
            forecast_horizon=self.forecast_horizon,
            mode=self.mode,
            cfg=self.config,
        )

    def sort_df(self, df):
        """
        Sort df rows and columns.
        """
        df = df.sort_values(by=[self.level, self.date])
        df = df[[self.level, self.date] + self.features + [self.power]]
        df = df.reset_index(drop=True)

        return df
