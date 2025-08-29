import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Aggregator:
    def __init__(
        self,
        level,
        granularity,
        date,
        power,
        hierarchies,
        aggr_dict,
    ):
        self.level = level
        self.granularity = granularity
        self.date = date
        self.power = power
        self.hierarchies = hierarchies
        self.aggr_dict = aggr_dict

    def aggregate_level_granularity(self, df) -> pd.DataFrame:
        """
        Aggregate to the given level, warehouse_id and the given time granularity.

        Parameters
        ----------
        df: pandas DataFrame

        Returns
        ----------
        df: pandas DataFrame, one row for each time point (depending on granularity and frequency), each locations and each group of the level, self.power containing the aggregated sales for each group.
        """
        logger.info(f"Aggregating to level {self.level} and granularity {self.granularity}.")

        # Get columns to group by
        key = [self.level]

        # Get columns to aggregate
        aggr_features = [self.power]

        # Add hierarchy levels above level
        level_index = self.hierarchies.index(self.level)
        higher_levels = self.hierarchies[:level_index]
        aggr_features.extend(higher_levels)

        # Build dictionary for aggregation
        higher_level_dict = dict()
        for ele in higher_levels:
            higher_level_dict[ele] = "first"
        self.aggr_dict.update(higher_level_dict)

        # Aggregate
        if not all(feat in self.aggr_dict.keys() for feat in aggr_features):
            raise ValueError(
                "There are features in your dataframe that are not in the aggregation dictionary, namely: "
                + str([feat for feat in aggr_features if feat not in self.aggr_dict.keys()])
            )

        def infer_granularity(index):
            if len(index) < 2:
                raise ValueError("Need at least 2 dates to infer granularity")

            # Calculate the differences between consecutive timestamps
            diffs = index.diff().dropna()

            # Find the most common difference
            most_common_diff = diffs.mode()[0]

            # Return the inferred frequency
            return most_common_diff

        aggr_dict_copy = self.aggr_dict.copy()

        data_granularity = infer_granularity(df[self.date])
        if data_granularity == pd.to_timedelta(self.granularity):
            aggr_dict_copy[self.power] = "ffill"

        del aggr_dict_copy[self.date]

        resampled_dfs = []
        for (level), group_df in df.groupby(key):
            group_df.set_index(self.date, inplace=True)
            resampled_df = group_df.resample(self.granularity, label="left").agg(aggr_dict_copy)
            resampled_df[self.level] = np.repeat(level, len(resampled_df))
            resampled_dfs.append(resampled_df)

        df = pd.concat(resampled_dfs)

        df = df.reset_index()

        df = df.sort_values(by=key)

        assert ~df.duplicated().any(), "The DataFrame has duplicated entries!"

        return df
