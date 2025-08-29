from typing import Callable, Union

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.util import forecast_start
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.transform.feature import (
    LastValueImputation,
    MissingValueImputation,
)
from gluonts.model.predictor import Predictor
from eile_forecast.models import BaseModel


class SeasonalAggregatePredictor(RepresentablePredictor):
    """
    Seasonal aggegate forecaster. (From gluonts v0.16.1)

    For each time series :math:`y`, this predictor produces a forecast
    :math:`\\tilde{y}(T+k) = f\big(y(T+k-h), y(T+k-2h), ...,
    y(T+k-mh)\big)`, where :math:`T` is the forecast time,
    :math:`k = 0, ...,` `prediction_length - 1`, :math:`m =`num_seasons`,
    :math:`h =`season_length` and :math:`f =`agg_fun`.

    If `prediction_length > season_length` :math:\times `num_seasons`, then the
    seasonal aggregate is repeated multiple times. If a time series is shorter
    than season_length` :math:\times `num_seasons`, then the `agg_fun` is
    applied to the full time series.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    season_length
        Seasonality used to make predictions. If this is an integer, then a
        fixed sesasonlity is applied; if this is a function, then it will be
        called on each given entry's ``freq`` attribute of the ``"start"``
        field, and the returned seasonality will be used.
    num_seasons
        Number of seasons to aggregate.
    agg_fun
        Aggregate function.
    imputation_method
        The imputation method to use in case of missing values.
        Defaults to :py:class:`LastValueImputation` which replaces each missing
        value with the last value that was not missing.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        season_length: Union[int, Callable],
        num_seasons: int,
        agg_fun: Callable = np.nanmean,
        imputation_method: MissingValueImputation = LastValueImputation(),
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        assert not isinstance(season_length, int) or season_length > 0, "The value of `season_length` should be > 0"

        assert isinstance(num_seasons, int) and num_seasons > 0, "The value of `num_seasons` should be > 0"

        self.prediction_length = prediction_length
        self.season_length = season_length
        self.num_seasons = num_seasons
        self.agg_fun = agg_fun
        self.imputation_method = imputation_method

    def predict_item(self, item: DataEntry) -> Forecast:
        if isinstance(self.season_length, int):
            season_length = self.season_length
        else:
            season_length = self.season_length(item["start"].freq)

        target = np.asarray(item[FieldName.TARGET], np.float32)
        len_ts = len(target)
        forecast_start_time = forecast_start(item)

        assert len_ts >= 1, "all time series should have at least one data point"

        if np.isnan(target).any():
            target = target.copy()
            target = self.imputation_method(target)

        if len_ts >= season_length * self.num_seasons:
            # `indices` here is a 2D array where each row collects indices
            # from one of the past seasons. The first row is identical to the
            # one in `seasonal_naive` and the subsequent rows are similar
            # except that the indices are taken from a different past season.
            indices = [
                [len_ts - (j + 1) * season_length + k % season_length for k in range(self.prediction_length)]
                for j in range(self.num_seasons)
            ]
            samples = self.agg_fun(target[indices], axis=0).reshape((1, self.prediction_length))
        else:
            samples = np.full(
                shape=(1, self.prediction_length),
                fill_value=self.agg_fun(target),
            )

        return SampleForecast(
            samples=samples,
            start_date=forecast_start_time,
            item_id=item.get("item_id", None),
            info=item.get("info", None),
        )


class SeasAgg(BaseModel):
    """
    Seasonal aggegate forecaster.

    For each time series :math:`y`, this predictor produces a forecast
    :math:`\\tilde{y}(T+k) = f\big(y(T+k-h), y(T+k-2h), ...,
    y(T+k-mh)\big)`, where :math:`T` is the forecast time,
    :math:`k = 0, ...,` `prediction_length - 1`, :math:`m =`num_seasons`,
    :math:`h =`season_length` and :math:`f =`agg_fun`.

    If `prediction_length > season_length` :math:\times `num_seasons`, then the
    seasonal aggregate is repeated multiple times. If a time series is shorter
    than season_length` :math:\times `num_seasons`, then the `agg_fun` is
    applied to the full time series.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    season_length
        Seasonality used to make predictions. If this is an integer, then a
        fixed sesasonlity is applied; if this is a function, then it will be
        called on each given entry's ``freq`` attribute of the ``"start"``
        field, and the returned seasonality will be used.
    num_seasons
        Number of seasons to aggregate.
    """

    def __init__(self, forecast_horizon, seas, train_len, **kwargs):
        self.name = type(self).__name__

        super().__init__(
            self.name,
            forecast_horizon=forecast_horizon,
            forecast_step=kwargs["forecast_step"],
        )

        num_seasons = min(4, (train_len / seas))

        self.predictor = SeasonalAggregatePredictor(
            prediction_length=forecast_horizon, season_length=seas, num_seasons=num_seasons
        )

    def train(self, *args) -> Predictor:
        return self.predictor

    def predict(self, test_pairs_input, **kwargs):
        forecasts = self.predictor.predict(test_pairs_input)
        if not isinstance(forecasts, list):
            forecasts = list(forecasts)
        return forecasts
