import click
from zenml.integrations.mlflow.mlflow_environment import MLFLOW_ENVIRONMENT_NAME
from zenml.integrations.mlflow.steps import MLFlowDeployerConfig

from evaluating import keras_evaluator
from importing import LoadSpectrogramConfig, get_paths_by_file, dvc_load_spectrograms, dvc_cli_load_spectrograms, \
    feast_load_spectrograms
from deployment import deployment_trigger, DeploymentTriggerConfig, model_deployer
from training import LSTMConfig, lstm_trainer
from zenpipeline import dvc_train_evaluate_and_deploy_pipeline, dvc_train_evaluate_pipeline, \
    feast_train_evaluate_pipeline
from zenml.services import load_last_service_from_step
from zenml.environment import Environment


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


def run_dvc_library_pipeline(epochs: int, batch_size: int, optimizer: str, loss: str):
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


def run_feast_pipeline(epochs: int, batch_size: int, optimizer: str, loss: str):
    pipeline = feast_train_evaluate_pipeline(
        feast_load_spectrograms=feast_load_spectrograms(config=LoadSpectrogramConfig(max_timesteps=200)),
        lstm_trainer=lstm_trainer(config=LSTMConfig(
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            loss=loss,
        )),
        keras_evaluator=keras_evaluator()
    )

    pipeline.run()


def run_dvc_cli_pipeline(epochs: int, batch_size: int, optimizer: str, loss: str):
    deployment = dvc_train_evaluate_pipeline(
        get_paths_by_file=get_paths_by_file(),
        dvc_load_spectrograms=dvc_cli_load_spectrograms(config=LoadSpectrogramConfig(max_timesteps=200)),
        lstm_trainer=lstm_trainer(config=LSTMConfig(
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            loss=loss,
        )),
        keras_evaluator=keras_evaluator()
    )

    deployment.run()


@click.command()
@click.option("--epochs", default=50, help="Number of epochs for training")
@click.option("--batch_size", default=10, help="Batch size for training")
@click.option("--optimizer", default="adam", help="optimizer for training")
@click.option("--loss", default="binary_crossentropy", help="loss function for training")
@click.option(
    "--stop-tensorboard",
    is_flag=True,
    default=False,
    help="Stop the Tensorboard server",
)
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
@click.option(
    "--min-accuracy",
    default=0.00,
    help="Minimum accuracy required to deploy the model",
)
@click.option(
    "--importing",
    default="dvc",
    help="Method for fetching training data: dvc, dvc_library, dvc_cli or feast",
)
def main(
    epochs: int,
    batch_size: int,
    optimizer: str,
    loss: str,
    stop_tensorboard: bool,
    stop_service: bool,
    min_accuracy: float,
    importing: str,
):
    if stop_service:
        service = load_last_service_from_step(
            pipeline_name="dvc_train_evaluate_and_deploy_pipeline",
            step_name="model_deployer",
            running=True,
        )
        if service:
            service.stop(timeout=10)
        return

    if importing == "dvc_library":
        run_dvc_library_pipeline(epochs=epochs, batch_size=batch_size, optimizer=optimizer, loss=loss)

        mlflow_env = Environment()[MLFLOW_ENVIRONMENT_NAME]
        print(
            "You can run:\n "
            f"[italic green]    mlflow ui --backend-store-uri {mlflow_env.tracking_uri} -p 4040[/italic green]\n"
            "...to inspect your experiment runs within the MLflow UI.\n"
            "You can find your runs tracked within the `mlflow_example_pipeline`"
            "experiment. There you'll also be able to compare two or more runs.\n\n"
        )
    elif importing == "dvc" or importing == 'dvc_cli':
        run_dvc_cli_pipeline(epochs=epochs, batch_size=batch_size, optimizer=optimizer, loss=loss)
        mlflow_env = Environment()[MLFLOW_ENVIRONMENT_NAME]
        print(
            "You can run:\n "
            f"[italic green]    mlflow ui --backend-store-uri {mlflow_env.tracking_uri} -p 4040[/italic green]\n"
            "...to inspect your experiment runs within the MLflow UI.\n"
            "You can find your runs tracked within the `mlflow_example_pipeline`"
            "experiment. There you'll also be able to compare two or more runs.\n\n"
        )
    elif importing == "feast":
        run_feast_pipeline(epochs=epochs, batch_size=batch_size, optimizer=optimizer, loss=loss)
    else:
        raise ValueError(f"{importing} is not a valid importing method, expected dvc, dvc_cli or feast")

    service = load_last_service_from_step(
        pipeline_name="dvc_train_evaluate_and_deploy_pipeline",
        step_name="model_deployer",
        running=True,
    )
    if service:
        print(
            f"The MLflow prediction server is running locally as a daemon process "
            f"and accepts inference requests at:\n"
            f"    {service.prediction_uri}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )

if __name__ == "__main__":
    main()
