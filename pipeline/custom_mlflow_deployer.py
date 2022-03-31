#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from typing import Optional, Type, cast

from mlflow import get_artifact_uri  # type: ignore[import]
from mlflow.tracking import MlflowClient  # type: ignore[import]

from zenml.environment import Environment
from zenml.integrations.mlflow.mlflow_environment import (
    MLFLOW_STEP_ENVIRONMENT_NAME,
    MLFlowStepEnvironment,
)
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.integrations.mlflow.services.mlflow_deployment import (
    MLFlowDeploymentConfig,
    MLFlowDeploymentService,
)
from zenml.integrations.mlflow.steps import MLFlowDeployerConfig
from zenml.services import load_last_service_from_step
from zenml.steps import (
    STEP_ENVIRONMENT_NAME,
    BaseStep,
    BaseStepConfig,
    StepContext,
    StepEnvironment,
    step,
)


@enable_mlflow
@step
def custom_mlflow_model_deployer(
    deploy_decision: bool,
    path: str,
    config: MLFlowDeployerConfig,
    context: StepContext,
) -> MLFlowDeploymentService:
    """MLflow model deployer pipeline step

    Args:
        deploy_decision: whether to deploy the model or not
        config: configuration for the deployer step
        context: pipeline step context

    Returns:
        MLflow deployment service
    """

    print(path)

    if not deploy_decision:
        return None

    # create a new service for the new model
    predictor_cfg = MLFlowDeploymentConfig(
        model_name=config.model_name,
        model_uri=path,
        workers=config.workers,
        mlserver=config.mlserver,
    )
    service = MLFlowDeploymentService(predictor_cfg)
    service.start(timeout=10)

    return service
