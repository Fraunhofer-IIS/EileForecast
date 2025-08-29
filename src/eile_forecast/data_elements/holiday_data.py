from typing import Optional
from datetime import datetime
import holidays
import numpy as np
import pandas as pd
from eile_forecast.data_elements.data import Data


class HolidayData(Data):
    def generate_data(self, startdate: datetime, enddate: datetime) -> pd.DataFrame:
        cfg = self.get_config()

        assert (
            "country" in cfg.datasets
        ), "For holiday data, you need to provide at least a country in dataset.yaml, better a country and a state."

        if "state" not in cfg.datasets:
            state = None
        else:
            state = cfg.datasets.state

        df = self.generate_holiday_and_weekend_dataframe(
            granularity="1h",
            country=cfg.datasets.country,
            state=state,
            startdate=startdate,
            enddate=enddate,
        )
        return df

    def generate_holiday_and_weekend_dataframe(
        self, granularity: str, country: str, state: Optional[str], startdate: datetime, enddate: datetime
    ) -> pd.DataFrame:

        # Feiertage für das angegebene Land abrufen
        if state is not None:
            state_holidays = holidays.country_holidays(country, state)
        else:
            state_holidays = holidays.country_holidays(country)

        # granularity time stamps
        date_range = pd.date_range(startdate, enddate, freq=granularity)
        df = pd.DataFrame(date_range, columns=["date"])

        # create a holiday mask
        holiday_mask = np.array([date_time in state_holidays for date_time in df["date"]])
        # create a weekend mask -> check if each day is on a weekend (Saturday or Sunday)
        # 5 and 6 represent Saturday and Sunday respectively
        weekend_mask = np.array((df["date"].dt.dayofweek >= 5))
        # create a (holiday or weekend) mask
        holiday_or_weekend_mask = holiday_mask | weekend_mask

        # convert masks from bool to int and add columns to dataframe
        df["holiday"] = holiday_mask.astype(int)
        df["weekend"] = weekend_mask.astype(int)
        df["holiday_or_weekend"] = holiday_or_weekend_mask.astype(int)

        return df
