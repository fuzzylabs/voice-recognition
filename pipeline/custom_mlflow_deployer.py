
from mlflow import get_artifact_uri  # type: ignore[import]
from mlflow.tracking import MlflowClient  # type: ignore[import]

from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.integrations.mlflow.services.mlflow_deployment import (
    MLFlowDeploymentConfig,
    MLFlowDeploymentService,
)
from zenml.integrations.mlflow.steps import MLFlowDeployerConfig
from zenml.steps import (
    StepContext,
    step,
)


@enable_mlflow
@step
def custom_mlflow_model_deployer(
    deploy_decision: bool,
    path: str,
    config: MLFlowDeployerConfig,
) -> MLFlowDeploymentService:
    """MLflow model deployer pipeline step

    Args:
        deploy_decision: whether to deploy the model or not
        config: configuration for the deployer step
        config: MLFlow deployment config

    Returns:
        MLflow deployment service
    """

    # We print the path so that the user knows where to copy AudioClassifier.py to
    print(path)

    # The deployer step is redundant from this point as the deployment is missing a dependency on the AudioClassifier.py
    # file, to actually deploy the service:
    # - run the pipeline
    # - Copy AudioClassifier.py to the path printed above
    # - run ` mlflow models serve -m . --no-conda`in the path directory printed above

    return None

    # if not deploy_decision:
    #     return None
    #
    # # create a new service for the new model
    # predictor_cfg = MLFlowDeploymentConfig(
    #     model_name=config.model_name,
    #     model_uri=path,
    #     workers=config.workers,
    #     mlserver=config.mlserver,
    # )
    # service = MLFlowDeploymentService(predictor_cfg)
    # service.start(timeout=10)
    #
    # return service
