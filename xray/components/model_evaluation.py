import sys

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader

from xray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelEvaluationConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import Net


class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact
        self.device = model_evaluation_config.device

    def _load_model(self) -> Module:
        model = Net().to(self.device)

        model.load_state_dict(
            torch.load(
                self.model_trainer_artifact.trained_model_path,
                map_location=self.device,
            )
        )

        model.eval()
        return model

    def evaluate(self) -> float:
        try:
            logging.info("Starting model evaluation")

            test_loader: DataLoader = (
                self.data_transformation_artifact.test_dataloader
            )

            model: Module = self._load_model()

            total_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)
                    loss = F.nll_loss(outputs, labels, reduction="sum")

                    total_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)

            avg_loss = total_loss / total
            accuracy = 100.0 * correct / total

            logging.info(
                f"Evaluation completed | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%"
            )

            return accuracy

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            accuracy = self.evaluate()

            return ModelEvaluationArtifact(model_accuracy=accuracy)

        except Exception as e:
            raise XRayException(e, sys)
