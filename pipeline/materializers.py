import logging
import os
from typing import Type

import keras
import mlflow
from zenml.artifacts import DataArtifact
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from training import AudioClassifier


class AudioClassifierMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (AudioClassifier,)
    ASSOCIATED_ARTIFACT_TYPES = (DataArtifact,)

    def _get_uri(self, name: str):
        return os.path.join(self.artifact.uri, 'audio_classifier', name)

    def handle_input(self, data_type: Type[AudioClassifier]) -> AudioClassifier:
        """Read from artifact store"""
        super().handle_input(data_type)

        with fileio.open(self._get_uri("timesteps.txt"), 'r') as f:
            timesteps = f.read()

        logging.getLogger("Hello World!").debug(f"Loading from Artifact store: {self._get_uri('')}")

        return AudioClassifier(mlflow.keras.load_model(self._get_uri("model")), int(timesteps))
        # return AudioClassifier(keras.models.load_model(self._get_model_uri(self.artifact.uri)))

    def handle_return(self, model: AudioClassifier) -> None:
        """Write to artifact store"""

        logging.getLogger("Hello World!").debug(f"Writing to Artifact store: {self._get_uri('')}")

        super().handle_return(model)
        mlflow.keras.save_model(model.model, self._get_uri("model"))
        with fileio.open(self._get_uri("timesteps.txt"), 'w') as f:
            f.write(str(model.timesteps))

