
import numpy as np
import pandas as pd

from zenml.steps import step, Output, BaseStepConfig

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout


class LSTMConfig(BaseStepConfig):
    """LSTM params"""
    epochs: int = 50
    batch_size: int = 10
    optimizer: str = "adam"
    loss: str = "mean_squared_error"


@step
def lstm_trainer(
    config: LSTMConfig,  # not an artifact; used for quickly changing params in runs
    X_train: np.ndarray,
    y_train: np.ndarray,
    timesteps: int,
) -> Output(
    model=Model, losses=np.ndarray, accuracies=np.ndarray
):
    """Train a LSTM to tell the difference between hello and goodbye spectrograms"""
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=False, input_shape=(timesteps, 1025)))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size)

    return model, np.array(history.history["loss"]), np.array(history.history["accuracy"])
