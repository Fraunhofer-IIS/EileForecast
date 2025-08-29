from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.model.predictor import Predictor
from eile_forecast.models import BaseModel


class SeasNaive(BaseModel):
    """
    Seasonal naïve forecaster.

    For each time series :math:`y`, this predictor produces a forecast
    :math:`\\tilde{y}(T+k) = y(T+k-h)`, where :math:`T` is the forecast time,
    :math:`k = 0, ...,` `forecast_horizon - 1`, and :math:`h =`
    `season_length`.

    If `forecast_horizon > season_length`, then the season is repeated
    multiple times. If a time series is shorter than season_length, then the
    mean observed value is used as prediction.

    Parameters
    ----------
    forecast_horizon
        Number of time points to predict
    seas
        Length of the seasonality pattern of the input data
    """

    def __init__(self, forecast_horizon, seas, **kwargs):
        self.name = type(self).__name__ + str(seas)
        super().__init__(
            self.name,
            forecast_horizon=forecast_horizon,
            forecast_step=kwargs["forecast_step"],
        )

        self.predictor = SeasonalNaivePredictor(
            freq="1h",
            prediction_length=forecast_horizon,
            season_length=seas,
        )

    def train(self, *args) -> Predictor:
        return self.predictor

    def predict(self, test_pairs_input, **kwargs):
        forecasts = self.predictor.predict(test_pairs_input)
        if not isinstance(forecasts, list):
            forecasts = list(forecasts)
        return forecasts
