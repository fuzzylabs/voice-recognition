import os

import keras.engine.training
import numpy as np
import mlflow
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow

from zenml.steps import step, BaseStepConfig, StepContext

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import TensorBoard

from importing import spectrogram_from_file, prep_spectrogram


class LSTMConfig(BaseStepConfig):
    """LSTM params"""
    epochs: int = 50
    batch_size: int = 10
    optimizer: str = "adam"
    loss: str = "mean_squared_error"


# Define the step and enable MLflow (n.b. order of decorators is important here)
@enable_mlflow
@step(enable_cache=True)
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

    mlflow.tensorflow.autolog()
    model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size, callbacks=[tensorboard_callback])

    return model


class AudioClassifier(mlflow.pyfunc.PythonModel):

    model: keras.engine.training.Model

    def predict(self, context, model_input: str):
        # Predict takes a byte encoded wav, converts it to a spectrogram and returns the models prediction
        # return self.model.predict(
        #     wav_bytes_to_spectrogram(base64.b64decode(model_input)),
        #     self.timesteps
        # )
        return prep_spectrogram(spectrogram_from_file("audio/hello/1.wav"), new_timesteps=200)
        # return np.array([[1, 2, 3], [4, 5, 6]])

    def __init__(self, model: keras.engine.training.Model, timesteps: int):
        self.model = model
        self.timesteps = timesteps


# Define the step and enable MLflow (n.b. order of decorators is important here)
@enable_mlflow
@step(enable_cache=True)
def lstm_trainer_preprocessing(
    config: LSTMConfig,  # not an artifact; used for quickly changing params in runs
    X_train: np.ndarray,
    y_train: np.ndarray,
    context: StepContext,
    timesteps: int,
) -> AudioClassifier:
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

    mlflow.tensorflow.autolog()
    model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size, callbacks=[tensorboard_callback])

    return AudioClassifier(model, timesteps=timesteps)
