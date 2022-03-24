import click
from zenml.integrations.mlflow.mlflow_environment import MLFLOW_ENVIRONMENT_NAME
from zenml.integrations.mlflow.steps import MLFlowDeployerConfig
from zenml.logger import set_root_verbosity

from evaluating import audio_classifier_evaluator
from importing import get_words, load_spectrograms_from_audio, LoadSpectrogramConfig
from deployment import deployment_trigger, DeploymentTriggerConfig, model_deployer, prediction_service_loader, \
    MLFlowDeploymentLoaderStepConfig
from materializers import AudioClassifierMaterializer
from training import LSTMConfig, lstm_trainer, lstm_trainer_preprocessing
from zenpipeline import train_evaluate_and_deploy_pipeline
from zenml.services import load_last_service_from_step
from zenml.environment import Environment

from zenml.integrations.tensorflow.visualizers import (
    visualize_tensorboard,
    stop_tensorboard_server,
)


@click.command()
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
def main(stop_tensorboard: bool, min_accuracy: float, stop_service: bool):
    set_root_verbosity()

    if stop_tensorboard:
        stop_tensorboard_server(
            pipeline_name="train_evaluate_and_deploy_pipeline",
            step_name="lstm_trainer",
        )

    if stop_service:
        service = load_last_service_from_step(
            pipeline_name="train_evaluate_and_deploy_pipeline",
            step_name="model_deployer",
            running=True,
        )
        if service:
            service.stop(timeout=10)

    if stop_tensorboard or stop_service:
        return

    deployment = train_evaluate_and_deploy_pipeline(
        get_words=get_words(),
        spectrogram_producer=load_spectrograms_from_audio(
            config=LoadSpectrogramConfig()
        ),
        lstm_trainer=lstm_trainer_preprocessing(config=LSTMConfig(
            epochs=1,
            batch_size=10,
            optimizer="adam",
            loss="mean_squared_error",
        )).with_return_materializers(AudioClassifierMaterializer),
        keras_evaluator=audio_classifier_evaluator(),
        deployment_trigger=deployment_trigger(
            config=DeploymentTriggerConfig(
                min_accuracy=min_accuracy,
            )
        ),
        model_deployer=model_deployer(config=MLFlowDeployerConfig(workers=3)),
    )

    deployment.run()

    # visualize_tensorboard(
    #     pipeline_name="train_evaluate_and_deploy_pipeline",
    #     step_name="lstm_trainer",
    # )
    #
    # Initialize an inference pipeline run
    # inference = inference_pipeline(
    #     get_words=get_words(),
    #     spectrogram_producer=load_spectrograms_from_audio(),
    #     prediction_service_loader=prediction_service_loader(
    #         MLFlowDeploymentLoaderStepConfig(
    #             pipeline_name="train_evaluate_and_deploy_pipeline",
    #             step_name="model_deployer",
    #         )
    #     ),
    #     predictor=predictor(),
    # )
    #
    # inference.run()

    mlflow_env = Environment()[MLFLOW_ENVIRONMENT_NAME]
    print(
        "You can run:\n "
        f"[italic green]    mlflow ui --backend-store-uri {mlflow_env.tracking_uri} -p 4040[/italic green]\n"
        "...to inspect your experiment runs within the MLflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. There you'll also be able to compare two or more runs.\n\n"
    )

    service = load_last_service_from_step(
        pipeline_name="train_evaluate_and_deploy_pipeline",
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
