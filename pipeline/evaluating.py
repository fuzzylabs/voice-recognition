
import numpy as np
import tensorflow as tf

from zenml.steps import step, Output


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


@step
def keras_hparam_evaluator(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: tf.keras.Model,
) -> Output(
    loss=float, accuracy=float
):
    """Calculate the loss for the model on the test set"""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    tf.summary.scalar("accuracy", accuracy, step=1)
    tf.summary.scalar("loss", loss, step=1)
    return
