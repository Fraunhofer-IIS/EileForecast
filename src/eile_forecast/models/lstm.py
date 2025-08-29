import os
import numpy as np
import pandas as pd
from keras.models import Sequential
import keras
from keras.callbacks import EarlyStopping
from keras.models import load_model
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import FileDataset
from eile_forecast.models import BaseModel


class LSTM(BaseModel):

    def __init__(self, forecast_horizon, max_epochs, early_stopping_patience, **kwargs):
        self.name = type(self).__name__
        super().__init__(
            self.name,
            forecast_horizon=forecast_horizon,
            forecast_step=kwargs["forecast_step"],
            past_horizon=kwargs["past_horizon"],
        )
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.model = None

    def create_dataset(self, target, external, past_horizon=1, forecast_horizon=1):
        X, y = [], []
        for i in range(len(target) - past_horizon - forecast_horizon + 1):
            X.append(
                np.column_stack(
                    (
                        target[i : (i + past_horizon)],
                        external[i + forecast_horizon : (i + past_horizon + forecast_horizon)],
                    )
                )
            )
            y.append(target[i + past_horizon : i + past_horizon + forecast_horizon])
        return np.array(X), np.array(y)

    def train(self, dataset: FileDataset) -> Predictor:
        model_path = "predictor.h5"

        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            y_train = dataset[1]
            x_train = dataset[0]
            x_train_cleaned = x_train.dropna()
            y_train_cleaned = y_train.loc[x_train_cleaned.index]

            x, y = self.create_dataset(
                y_train_cleaned, x_train_cleaned, past_horizon=self.past_horizon, forecast_horizon=self.forecast_horizon
            )

            # reshape data for LSTM [Samples, Time Steps, Features]
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))

            # Split data into training and validation sets
            split_index = int(0.8 * len(x))  # 80% training, 20% validation
            x_train, x_val = x[:split_index], x[split_index:]
            y_train, y_val = y[:split_index], y[split_index:]

            self.model = Sequential()
            self.model.add(keras.layers.LSTM(50, input_shape=(x.shape[1], x.shape[2])))
            self.model.add(keras.layers.Dense(self.forecast_horizon))
            self.model.compile(optimizer="adam", loss="mean_squared_error")

            early_stopping = EarlyStopping(
                monitor="val_loss", patience=self.early_stopping_patience, restore_best_weights=True
            )

            # Fit the model with validation data
            self.model.fit(
                x_train,
                y_train,
                epochs=self.max_epochs,
                verbose=1,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping],
            )
            self.model.save("predictor.h5")

        return self.model

    def predict(self, test_pairs_input):
        x_test = test_pairs_input[0]
        y_test = test_pairs_input[1]
        x_train = test_pairs_input[2]
        y_train = test_pairs_input[3]
        x_input = pd.concat([x_train, x_test])

        # prepare input data for forecast
        input_data = np.column_stack((y_train[-self.past_horizon :], x_input[-self.past_horizon :])).reshape(
            (1, self.past_horizon, x_test.shape[1] + 1)
        )
        prediction = self.model.predict(input_data)

        prediction[prediction < 0] = 0  # negative power is physically impossible
        pred_df = y_test.copy()
        pred_df["pred"] = prediction.flatten()
        pred_df.drop(columns={"power"}, inplace=True)
        pred_df.reset_index(inplace=True)
        pred_df.rename(columns={"pred": "sample_fcst0", "date": "fcst_step_date", "signal_id": "item_id"}, inplace=True)

        gluonts_forecast = self.table_forecast_to_gluonts(df=pred_df)
        return gluonts_forecast
