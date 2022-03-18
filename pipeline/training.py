import os

import numpy as np

from zenml.steps import step, BaseStepConfig, StepContext

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import TensorBoard
from keras.callbacks import CallbackList
from tensorboard.plugins.hparams import api as hp


class LSTMConfig(BaseStepConfig):
    """LSTM params"""
    epochs: int = 50
    batch_size: int = 10
    optimizer: str = "adam"
    loss: str = "mean_squared_error"


@step(enable_cache=False)
def lstm_trainer(
    config: LSTMConfig,  # not an artifact; used for quickly changing params in runs
    X_train: np.ndarray,
    y_train: np.ndarray,
    context: StepContext,
    timesteps: int,
) -> Model:
    """Train a LSTM to tell the difference between hello and goodbye spectrograms"""
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=False, input_shape=(timesteps, 1025)))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=["accuracy"])

    log_dir = os.path.join(context.get_output_artifact_uri(), "logs")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size, callbacks=[tensorboard_callback])

    return model

@step(enable_cache=True)
def lstm_hparam_trainer(
    config: LSTMConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    context: StepContext,
    timesteps: int,
    callbacks: CallbackList
) -> Model:
    """Train a LSTM to tell the difference between hello and goodbye spectrograms"""
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=False, input_shape=(timesteps, 1025)))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=["accuracy"])

    log_dir = os.path.join(context.get_output_artifact_uri(), "logs")
    model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size, callbacks=[
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        hp.KerasCallback(log_dir, {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "optimizer": config.optimizer,
            "loss": config.loss,
        }),
    ])

    return model
