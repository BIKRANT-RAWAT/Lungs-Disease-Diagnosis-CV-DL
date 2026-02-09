import os
import sys

import bentoml
import joblib
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from xray.constant.training_pipeline import *
from xray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelTrainerConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import Net


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.device = model_trainer_config.device

        self.model: Module = Net().to(self.device)

    def _train_one_epoch(self, optimizer: Optimizer) -> float:
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.data_transformation_artifact.train_dataloader)

        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()

            output = self.model(data)
            loss = F.nll_loss(output, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            pbar.set_description(
                f"Loss={loss.item():.4f} Acc={100*correct/total:.2f}%"
            )

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def _evaluate(self) -> float:
        self.model.eval()

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.data_transformation_artifact.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = F.nll_loss(output, target, reduction="sum")

                test_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        avg_loss = test_loss / total
        accuracy = 100.0 * correct / total

        logging.info(
            f"Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%"
        )

        return accuracy

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")

            optimizer = torch.optim.SGD(
                self.model.parameters(),
                **self.model_trainer_config.optimizer_params,
            )

            scheduler = StepLR(
                optimizer, **self.model_trainer_config.scheduler_params
            )

            best_accuracy = 0.0

            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)

            for epoch in range(1, self.model_trainer_config.epochs + 1):
                logging.info(f"Epoch {epoch}/{self.model_trainer_config.epochs}")

                train_loss, train_acc = self._train_one_epoch(optimizer)
                test_acc = self._evaluate()

                scheduler.step()

                if test_acc > best_accuracy:
                    best_accuracy = test_acc

                    torch.save(
                        self.model.state_dict(),
                        self.model_trainer_config.trained_model_path,
                    )

                    logging.info(
                        f"New best model saved with accuracy: {best_accuracy:.2f}%"
                    )

            # Load best model for packaging
            self.model.load_state_dict(
                torch.load(self.model_trainer_config.trained_model_path)
            )

            train_transform = joblib.load(
                self.data_transformation_artifact.train_transform_file_path
            )

            bentoml.pytorch.save_model(
                name=self.model_trainer_config.trained_bentoml_model_name,
                model=self.model,
                custom_objects={"transform": train_transform},
            )

            return ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path
            )

        except Exception as e:
            raise XRayException(e, sys)
