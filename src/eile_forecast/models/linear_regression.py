import sklearn.linear_model
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import FileDataset
from eile_forecast.models import BaseModel


class LinearRegression(BaseModel):

    def __init__(self, forecast_horizon, **kwargs):
        self.name = type(self).__name__
        super().__init__(
            self.name,
            forecast_horizon=forecast_horizon,
            forecast_step=kwargs["forecast_step"],
            past_horizon=kwargs["past_horizon"],
        )

        self.estimator = sklearn.linear_model.LinearRegression()

    def train(self, dataset: FileDataset) -> Predictor:
        x_train = dataset[0]
        y_train = dataset[1]
        x_train_cleaned = x_train.dropna()
        y_train_cleaned = y_train.loc[x_train_cleaned.index]
        self.predictor = self.estimator.fit(x_train_cleaned, y_train_cleaned)
        return self.predictor

    def predict(self, test_pairs_input):
        x_test = test_pairs_input[0]
        y_test = test_pairs_input[1]
        prediction = self.predictor.predict(x_test.to_numpy())
        prediction[prediction < 0] = 0  # negative power is physically impossible
        pred_df = y_test.copy()
        pred_df["pred"] = prediction
        pred_df.drop(columns={"power"}, inplace=True)
        pred_df.reset_index(inplace=True)
        pred_df.rename(columns={"pred": "sample_fcst0", "date": "fcst_step_date", "signal_id": "item_id"}, inplace=True)

        gluonts_forecast = self.table_forecast_to_gluonts(df=pred_df)
        return gluonts_forecast
