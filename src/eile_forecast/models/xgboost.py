import logging
from typing import Any
import yaml
from pandas import DataFrame
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from eile_forecast.models import BaseModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

xgb.set_config(verbosity=0)


class XGboost(BaseModel):
    def __init__(
        self,
        learning_rate: float,
        max_depth: int,
        gamma,
        reg_alpha,
        reg_lambda,
        colsample_bytree,
        min_child_weight,
        n_estimators,
        forecast_horizon: int,
        forecast_step: int,
        early_stopping_rounds,
        path_to_configs: str,
        name: str = "xgboost",
        hyperparameter_tuning: bool = False,
        **kwargs
    ) -> None:

        self.name = name
        self.path_to_configs = path_to_configs
        super().__init__(
            self.name,
            forecast_horizon,
            forecast_step=forecast_step,
            past_horizon=kwargs["past_horizon"],
        )

        keys = [
            "learning_rate",
            "max_depth",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "colsample_bytree",
            "min_child_weight",
            "n_estimators",
        ]

        values = [
            learning_rate,
            max_depth,
            gamma,
            reg_alpha,
            reg_lambda,
            colsample_bytree,
            min_child_weight,
            n_estimators,
        ]

        self.params = dict(map(lambda i, j: (i, j), keys, values))

        self.early_stopping = early_stopping_rounds
        scaler = StandardScaler()
        xgboostreg = XGBRegressor(**self.params, tree_method="hist", verbosity=0, silent=1)

        self.model = Pipeline(steps=[("scaler", scaler), ("xgb", xgboostreg)])
        self.name = name
        self.forecast_horizon = forecast_horizon
        self.forecast_step = forecast_step
        self.hyperparameter_tuning = hyperparameter_tuning
        self.predictor = None

    def train(self, dataset: list) -> Any:

        x_train = dataset[0]
        y_train = dataset[1]
        x_val = dataset[2]
        y_val = dataset[3]

        if self.hyperparameter_tuning:
            # hyperparameter optimization (needs to be done only once)
            param_dist = {
                "max_depth": randint(3, 25),
                "learning_rate": uniform(0.01, 0.5),
                "n_estimators": randint(100, 500),
                "subsample": uniform(0.6, 0.4),
                "gamma": uniform(0, 20),
                "reg_alpha": uniform(0, 100),
                "reg_lambda": uniform(0, 100),
                "colsample_bytree": uniform(0.6, 0.4),
                "min_child_weight": uniform(0, 10),
            }

            # Configure the XGBoost model
            xgb_model = xgb.XGBRegressor()
            # Set up the randomized search with cross-validation
            random_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_dist,
                n_iter=100,
                scoring="neg_mean_squared_error",
                cv=3,
                verbose=1,
                random_state=42,
            )

            # Fit the randomized search model
            random_search.fit(x_val, y_val)
            # Get the best parameters and best score
            logger.info("Hyperparameters fitted. Best parameters:{}".format(random_search.best_params_))
            # write the best parameters into the model config file
            with open(self.path_to_configs + "models/XGboost.yaml", "r") as file:
                config = yaml.safe_load(file)

            config["colsample_bytree"] = float(random_search.best_params_["colsample_bytree"])
            config["gamma"] = float(random_search.best_params_["gamma"])
            config["learning_rate"] = float(random_search.best_params_["learning_rate"])
            config["max_depth"] = int(random_search.best_params_["max_depth"])
            config["min_child_weight"] = float(random_search.best_params_["min_child_weight"])
            config["n_estimators"] = int(random_search.best_params_["n_estimators"])
            config["reg_alpha"] = float(random_search.best_params_["reg_alpha"])
            config["reg_lambda"] = float(random_search.best_params_["reg_lambda"])
            config["hyperparameter_tuning"] = bool(False)
            with open(self.path_to_configs + "models/XGboost.yaml", "w") as file:
                yaml.dump(config, file)

        else:
            self.model.fit(x_train, y_train)

            self.predictor = self.model

        return None

    def predict(self, test_pairs_input: DataFrame) -> DataFrame:
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
