from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_deployer_step
from zenml.pipelines import pipeline
from zenml.services import load_last_service_from_step
from zenml.steps import step, Output, BaseStepConfig, StepContext
import numpy as np


class DeploymentTriggerConfig(BaseStepConfig):
    """Parameters that are used to trigger the deployment"""
    min_accuracy: float


@step
def deployment_trigger(
        accuracy: float,
        config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy > config.min_accuracy


class MLFlowDeploymentLoaderStepConfig(BaseStepConfig):
    """MLflow deployment getter configuration
    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
    """

    pipeline_name: str
    step_name: str
    running: bool = True


@step(enable_cache=False)
def prediction_service_loader(
        config: MLFlowDeploymentLoaderStepConfig, context: StepContext
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline"""

    service = load_last_service_from_step(
        pipeline_name=config.pipeline_name,
        step_name=config.step_name,
        step_context=context,
        running=config.running,
    )
    if not service:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{config.step_name} step in the {config.pipeline_name} pipeline "
            f"is currently running."
        )

    return service


@step
def predictor(
        service: MLFlowDeploymentService,
        data: np.ndarray,
) -> Output(predictions=np.ndarray):
    """Run a inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    prediction = service.predict(data)
    prediction = prediction.argmax(axis=-1)

    return prediction


# @pipeline(enable_cache=True, requirements_file="pipeline-requirements.txt")
# def inference_pipeline(
#         get_words,
#         spectrogram_producer,
#         prediction_service_loader,
#         predictor,
# ):
#     X_train, X_test, y_train, y_test, timesteps = spectrogram_producer(get_words())
#     model_deployment_service = prediction_service_loader()
#     predictor(model_deployment_service, X_test)


model_deployer = mlflow_deployer_step(name="model_deployer")
