import os
import numpy as np
import mlflow
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow

from zenml.steps import step, BaseStepConfig, StepContext

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import TensorBoard

from zenml.artifacts import ModelArtifact

from AudioClassifier import AudioClassifier


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


@enable_mlflow
@step
def attach_model_preprocessing(
    model: ModelArtifact,
    context: StepContext,
) -> str:
    # TODO: Pass the timestep value through the dictionary
    artifacts = {
        "model": model.uri
    }

    requirements_env = [
        'mlflow',
        'librosa',
        'soundfile',
        'numpy',
        'tensorflow',
    ]

    model_saved_path = os.path.join(context.get_output_artifact_uri(), "model")

    mlflow.pyfunc.save_model(
        path=model_saved_path, python_model=AudioClassifier(), artifacts=artifacts, pip_requirements=requirements_env,
    )

    return model_saved_path
