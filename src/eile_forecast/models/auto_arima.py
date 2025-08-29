import numpy as np
from gluonts.ext.statsforecast import AutoARIMAPredictor
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast
from eile_forecast.models import BaseModel


class AutoArima(BaseModel):
    """Statsforecast AutoArima wrapped by GluonTS.

    Args:
        forecast_horizon: forecast horizon.
        train_len: truncation length.
    """

    def __init__(self, forecast_horizon, train_len, **kwargs):
        self.name = type(self).__name__
        super().__init__(
            self.name,
            forecast_horizon=forecast_horizon,
            forecast_step=kwargs["forecast_step"],
        )

        self.estimator = AutoARIMAPredictor(
            prediction_length=forecast_horizon,
            truncate=train_len,
            season_length=24,
            nmodels=5,
            trace=True,  # period makes the model seasonal
        )
        self.predictor = None

    def train(self, *args) -> Predictor:
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
