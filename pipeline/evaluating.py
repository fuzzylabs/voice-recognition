
import numpy as np
import tensorflow as tf

from zenml.steps import step, Output
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow


# N.b: The order of decorators is important here
@enable_mlflow
@step
def keras_evaluator(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: tf.keras.Model,
) -> Output(
    loss=float, accuracy=float
):
    """Calculate the loss for the model on the test set"""
    return model.evaluate(X_test, y_test, verbose=2)
