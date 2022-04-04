import os

import numpy as np
import mlflow
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow

from zenml.steps import step, BaseStepConfig, StepContext

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import TensorBoard


class LSTMConfig(BaseStepConfig):
    """LSTM params"""
    epochs: int = 50
    batch_size: int = 10
    optimizer: str = "adam"
    loss: str = "binary_crossentropy"


# Define the step and enable MLflow (n.b. order of decorators is important here)
@enable_mlflow
@step(enable_cache=True)
def lstm_trainer(
    config: LSTMConfig,  # not an artifact; used for quickly changing params in runs
    X_train: np.ndarray,
    y_train: np.ndarray,
    timesteps: int,
) -> Model:
    """Train a LSTM to tell the difference between hello and goodbye spectrograms"""

    mlflow.log_param("Optimizer", config.optimizer)
    mlflow.log_param("Loss Function", config.loss)

    model = Sequential()

    model.add(LSTM(units=64, return_sequences=True, input_shape=(timesteps, 1025)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=32, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=16, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=2, activation="softmax"))

    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=["accuracy"])

    mlflow.tensorflow.autolog()
    model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size)

    return model
