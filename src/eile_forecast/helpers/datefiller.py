import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DateFiller:
    def __init__(self, date, granularity, level, hierarchies, power, features):
        self.date = date
        self.granularity = granularity
        self.level = level
        self.hierarchies = hierarchies
        self.power = power
        self.features = features
        self.idx = None
        self.df_first_dates = None

    def get_all_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill in rows for dates when there were no sales for level id and a specific locations,
        but only starting AFTER the first sale of this level id at this location. These rows will have power value 0.
        Constant columns are filled in.
        The returned df doesn't contain any nans, but includes all date x warehouse_id x level
        combinations where there could have been sales.
        Parameters
        ----------
        df: pd.DataFrame
        Returns
        ----------
        df: pd.DataFrame
        """
        logger.info("Filling in missing time stamps.")
        # Create time grid according to granularity and first an last timestamp
        self.get_global_date_range(df)
        # convert to dataTimeIndex
        df.index = pd.DatetimeIndex(df["date"])
        df.index.rename("date_idx", inplace=True)
        df = df.sort_values([self.level, "date_idx"])
        df = df.set_index([self.level], append=True)
        self.get_first_dates(df)
        # Create new index
        idx_df = self.get_multiindex(self.idx)
        df = df.reindex(index=idx_df)
        df = df.reset_index(level=[self.level])
        df = df.fillna(value={self.date: df.index.to_series()})
        # replace nans with last seen value
        df["level"] = df[self.level]
        df = df.groupby(self.level).ffill()
        df[self.level] = df["level"]
        df.drop(columns=["level", "date"], inplace=True)
        df.reset_index(inplace=True)
        if df.isna().any().any():
            columns_with_nans = df.columns[df.isna().any()].tolist()
            raise ValueError(f'The column(s) "{", ".join(columns_with_nans)}" still contain nans after replacing nans.')
        df.rename(columns={"date_idx": "date"}, inplace=True)
        return df

    def get_global_date_range(self, df):
        # calculate first and last timestamp
        first = df[self.date].min()
        last = df[self.date].max()

        self.idx = pd.date_range(first, last, freq=self.granularity)

    def get_first_dates(self, df):
        # get the starting sales dates for all the IDs in level
        self.df_first_dates = (
            df.drop(columns=[self.power] + self.hierarchies[: self.hierarchies.index(self.level)])
            .groupby([self.level], as_index=True)
            .min()
            .reset_index()
        )
        self.df_first_dates = self.df_first_dates.loc[:, [self.level, self.date]]
        self.df_first_dates.rename(columns={self.date: "first_date"}, inplace=True)

    def get_multiindex(self, idx: pd.date_range):
        """
        Create multiindex used to fill in zero-rows.
        For every level id, for every warehouse_id id it appears with, we want an index going from the first date that occurs for that combination to the last date in the data for all ids.

        Parameters
        ----------
        idx: pd.date_range
            Contains the date range across all the timestamps in the data.

        Returns
        ----------
        df: pd.DataFrame
            A dataframe containing the indices for all the level ids.

        """
        largest_date = max(idx)
        dfs = [
            pd.DataFrame(
                {
                    "date_idx": pd.date_range(row[1], largest_date, freq=self.granularity),
                    self.level: row[0],
                }
            )
            for row in self.df_first_dates.itertuples(index=False)
        ]
        return pd.concat(dfs, axis=0)
