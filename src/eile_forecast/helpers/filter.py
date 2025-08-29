class Filter:
    def __init__(self, signals, level, power):
        self.forecast_signal = signals
        self.level = level
        self.power = power

    def filter(self, df):
        df = self.filter_for_signal(df)
        return df

    def filter_for_signal(self, df):
        df = df[df["signal_id"] == self.forecast_signal]
        return df

    def filter_for_long_ts(self, df, min_len=4):
        """
        Filters for time series above a certain minimal length.

        Parameters
        ----------
        df: pd.DataFrame
        min_len: int
            The minimum length of the time series. Default equals 4 because that's what the GluonFableEstimator takes as minimum.

        Returns
        ----------
        df: pd.DataFrame
        """
        df_ts_len = df[[self.level, self.power]].groupby(by=[self.level]).count()
        df_ts_len = df_ts_len[df_ts_len[self.power] >= min_len]
        if df_ts_len.empty:
            raise ValueError("min_len is too large, no data left.")
        i1 = df_ts_len.index
        i2 = df.set_index([self.level]).index
        df = df[i2.isin(i1)]

        return df
