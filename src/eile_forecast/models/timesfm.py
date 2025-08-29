import os

os.environ["JAX_PLATFORMS"] = "cpu"
import math
import timesfm
from eile_forecast.models import BaseModel

# Loading the timesfm-2.0 checkpoint:
# For PAX


class TimesFM(BaseModel):
    """
    Pre-trained TimesFM model.

    Parameters
    ----------
    freq
        Frequency of the input data
    forecast_horizon
        Number of time points to predict
    """

    def __init__(self, forecast_horizon, train_len, **kwargs):
        self.name = type(self).__name__
        super().__init__(
            self.name,
            forecast_horizon=forecast_horizon,
            forecast_step=kwargs["forecast_step"],
        )

        context_len = math.floor(train_len / 32) * 32
        self.predictor = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,  # 32 is default
                horizon_len=forecast_horizon,
                num_layers=50,  # 50 is default
                context_len=context_len,  # needs to be a multiplier of 32!
                use_positional_embedding=False,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-jax"),
        )

    def train(self, *args):
        """
        Fine tune the model on the given dataset.

        """

        return self.predictor

    def predict(self, test_pairs_input, **kwargs):
        test_pairs_input[0].reset_index(inplace=True)
        test_pairs_input[0].rename(columns={"signal_id": "unique_id", "date": "ds"}, inplace=True)

        forecasts = self.predictor.forecast_on_df(
            inputs=test_pairs_input[0],
            freq="H",
            value_name="power",
            num_jobs=-1,
        )[["unique_id", "ds", "timesfm"]]

        forecasts.rename(
            columns={"unique_id": "item_id", "ds": "fcst_step_date", "timesfm": "sample_fcst0"}, inplace=True
        )

        gluonts_forecast = self.table_forecast_to_gluonts(df=forecasts)
        return gluonts_forecast
