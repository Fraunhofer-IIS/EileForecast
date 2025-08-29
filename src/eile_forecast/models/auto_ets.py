import numpy as np
from gluonts.ext.statsforecast import AutoETSPredictor
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast
from eile_forecast.models import BaseModel


class AutoETS(BaseModel):
    """Statsforecast AutoETS wrapped by GluonTS.

    Args:
        forecast_horizon: forecast horizon.
        season_len: season length.
    """

    def __init__(self, forecast_horizon, season_len, train_len, **kwargs):
        self.name = type(self).__name__
        super().__init__(
            self.name,
            forecast_horizon=forecast_horizon,
            forecast_step=kwargs["forecast_step"],
        )

        self.estimator = AutoETSPredictor(prediction_length=forecast_horizon, season_length=season_len)
        self.predictor = None

    def train(self, dataset) -> Predictor:
        self.predictor = self.estimator
        return self.predictor

    def predict(self, test_pairs_input, **kwargs):
        forecasts_quantiles = list(self.predictor.predict(test_pairs_input))
        first_forecast = forecasts_quantiles[0]
        samples = first_forecast["mean"]
        sample_paths = samples[np.newaxis, :]

        forecasts = [
            SampleForecast(
                samples=sample_paths,
                start_date=first_forecast.start_date,
                item_id=first_forecast.item_id,
            )
        ]
        return forecasts
