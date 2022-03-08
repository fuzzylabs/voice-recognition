
import numpy as np

from zenml.steps import step, BaseStepConfig
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout


class LSTMConfig(BaseStepConfig):
    """LSTM params"""
    epochs: int = 50
    batch_size: int = 10
    optimizer: str = "adam"
    loss: str = "mean_squared_error"


# N.b: The order of decorators is important here
@enable_mlflow
@step
def lstm_trainer(
    config: LSTMConfig,  # not an artifact; used for quickly changing params in runs
    X_train: np.ndarray,
    y_train: np.ndarray,
    timesteps: int,
) -> Model:
    """Train a LSTM to tell the difference between hello and goodbye spectrograms"""
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=False, input_shape=(timesteps, 1025)))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size)

    return model
