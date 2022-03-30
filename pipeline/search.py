from zenml.integrations.mlflow.steps import MLFlowDeployerConfig

from deployment import model_deployer, DeploymentTriggerConfig, deployment_trigger
from evaluating import keras_evaluator
from importing import dvc_load_spectrograms, LoadSpectrogramConfig, get_paths_by_file
from training import LSTMConfig, lstm_trainer
from zenpipeline import dvc_train_evaluate_and_deploy_pipeline, dvc_train_evaluate_pipeline


def run_deploy_pipeline(epochs: int, batch_size: int, optimizer: str, loss: str):
    deployment = dvc_train_evaluate_and_deploy_pipeline(
        get_paths_by_file=get_paths_by_file(),
        dvc_load_spectrograms=dvc_load_spectrograms(config=LoadSpectrogramConfig(max_timesteps=200)),
        lstm_trainer=lstm_trainer(config=LSTMConfig(
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            loss=loss,
        )),
        keras_evaluator=keras_evaluator(),
        deployment_trigger=deployment_trigger(
            config=DeploymentTriggerConfig(
                min_accuracy=1.0,
            )
        ),
        model_deployer=model_deployer(config=MLFlowDeployerConfig(workers=3)),
    )

    deployment.run()


def run_pipeline(epochs: int, batch_size: int, optimizer: str, loss: str):
    deployment = dvc_train_evaluate_pipeline(
        get_paths_by_file=get_paths_by_file(),
        dvc_load_spectrograms=dvc_load_spectrograms(config=LoadSpectrogramConfig(max_timesteps=200)),
        lstm_trainer=lstm_trainer(config=LSTMConfig(
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            loss=loss,
        )),
        keras_evaluator=keras_evaluator()
    )

    deployment.run()


if __name__ == "__main__":
    for epochs in range(5, 105, 5):
        for batch_size in range(5, 45, 5):
            run_pipeline(epochs=epochs, batch_size=batch_size, optimizer="adam", loss="mean_squared_error")
