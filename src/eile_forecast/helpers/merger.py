import numpy as np
import pandas as pd
from eile_forecast.data_elements.holiday_data import HolidayData


class Merger:
    def __init__(self, date, granularity, features, forecast_horizon, mode, cfg) -> None:
        """
        Class for merging additional features.

        """
        self.date = date
        self.granularity = granularity
        self.features = features
        self.forecast_horizon = forecast_horizon
        self.mode = mode
        self.cfg = cfg

    def merge_weather(self, df) -> pd.DataFrame:
        """
        Merge with past weather data in benchmark mode.
        Fill gaps with mean values of the respective hour.

        Returns
        ----------
        df: pandas DataFrame with additional weather data
        """
        if "raw_path" not in self.cfg.datasets:
            raw_path = None
        else:
            raw_path = self.cfg.datasets.raw_path
        assert raw_path is not None, "Please provide the raw path of the weather data in dataset.yaml"
        weather = pd.read_parquet(self.cfg.datasets.raw_path + "weather.parquet")
        weather.reset_index(inplace=True, drop=True)
        weather.rename(columns={"time": "date"}, inplace=True)

        weather.set_index(self.date, inplace=True)
        weather.index = pd.to_datetime(weather.index, utc=True)
        weather = weather.resample(self.granularity).asfreq()
        weather = weather.assign(hour=lambda x: x.index.hour)

        # fill missing values with mean daily values
        hr_pattern = weather.groupby("hour", as_index=False).mean()
        if "boolean_is_day" in hr_pattern.columns:
            hr_pattern["boolean_is_day"] = np.round(hr_pattern["boolean_is_day"]).astype(int)
        weather_hr_pattern = weather.merge(hr_pattern, on="hour", how="left")
        weather_hr_pattern.index = weather.index
        hour_index = weather_hr_pattern.columns.get_loc("hour")
        mean_day = weather_hr_pattern.iloc[:, hour_index + 1 :]
        mean_day.columns = mean_day.columns.str.replace("_y", "", regex=False)
        mean_day.index = weather.index[: len(mean_day)]
        weather.update(mean_day)

        weather.reset_index(inplace=True)
        df = pd.merge(left=df, right=weather, on="date")
        df.date = pd.to_datetime(df.date)
        self.features = set(self.features + [col for col in weather.columns if col not in ["date"]])
        return df

    def merge_holidays(self, df) -> pd.DataFrame:
        holidays_data = HolidayData(self.cfg)
        holidays = holidays_data.generate_data(startdate=df.date.min(), enddate=df.date.max())
        df = pd.merge(left=df, right=holidays, on="date")
        self.features = set(list(self.features) + ["holiday", "weekend", "holiday_or_weekend"])
        return df

    def get_periodicity(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """
        Takes in a dataframe and adds periodic seasonal components.

        Parameters
        ----------
        df: pandas DataFrame

        Returns
        ----------
        pd.DataFrame: dataframe with additional sine and cosine columns for daily, weekly, monthly, and yearly periodicity.
        list: list with features enlarged by those produced in the function
        """
        # f(t) = sin(2*pi*i*t/365.25), i -> Periodicity, t -> dates [1,2,3,4,...]
        date_min = min(df[self.date])
        date_max = max(df[self.date])

        df_dates = pd.DataFrame({"date": pd.date_range(start=date_min, end=date_max, freq=self.granularity)})
        df_dates["y_sine"] = df_dates["date"].apply(
            lambda x: np.sin(2 * np.pi * x.dayofyear / 366) if x.is_leap_year else np.sin(2 * np.pi * x.dayofyear / 365)
        )
        df_dates["y_cosine"] = df_dates["date"].apply(
            lambda x: np.cos(2 * np.pi * x.dayofyear / 366) if x.is_leap_year else np.cos(2 * np.pi * x.dayofyear / 365)
        )
        if self.granularity != "MS":
            df_dates["m_sine"] = df_dates["date"].apply(lambda x: np.sin(2 * np.pi * x.day / x.daysinmonth))
            df_dates["m_cosine"] = df_dates["date"].apply(lambda x: np.cos(2 * np.pi * x.day / x.daysinmonth))

            if not self.granularity.startswith("W"):
                df_dates["w_sine"] = df_dates["date"].apply(lambda x: np.sin(2 * np.pi * x.dayofweek / 7))
                df_dates["w_cosine"] = df_dates["date"].apply(lambda x: np.cos(2 * np.pi * x.dayofweek / 7))

                if self.granularity != "D":
                    df_dates["d_sine"] = df_dates["date"].apply(lambda x: np.sin(2 * np.pi * x.hour / 24))
                    df_dates["d_cosine"] = df_dates["date"].apply(lambda x: np.cos(2 * np.pi * x.hour / 24))

        df["Year"] = df[self.date].dt.year

        self.features = set(list(self.features) + [col for col in df_dates.columns if col not in ["date"]])

        df = df.merge(df_dates, left_on=self.date, right_on="date", how="left")

        return df, list(self.features)
