import keras
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
import matplotlib as plt
import seaborn as sns
from zenml.io import fileio

from zenml.steps import step, Output


def log_confusion_matrix(model: keras.Model, X_test: np.ndarray, y_true: np.ndarray):
    y_pred = model.predict(X_test)
    print(y_pred)
    print()
    print(y_true)
    print()
    conf_matrix = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    con_mat_norm = np.around(conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(
        con_mat_norm,
        index=["hello", "goodbye"],
        columns=["hello", "goodbye"],
    )

    print(con_mat_df)
    # fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    #
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    #
    # figure = plt.figure.Figure(figsize=(8, 8))
    # sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    # figure.tight_layout()
    # figure.add_axes(rect=[1,1,1,1], ylabel="True label", xlabel="Predicted label")
    # mlflow.log_figure(sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues), "confusion_matrix.png")


# Define the step and enable MLflow (n.b. order of decorators is important here)
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
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    mlflow.log_metric("Testing Loss", loss)
    mlflow.log_metric("Testing Accuracy", accuracy)

    log_confusion_matrix(model, X_test, y_test)

    return loss, accuracy
