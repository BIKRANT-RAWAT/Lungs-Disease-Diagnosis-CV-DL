import os
import sys
from typing import Tuple

import joblib
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from xray.entity.artifacts_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from xray.entity.config_entity import DataTransformationConfig
from xray.exception import XRayException
from xray.logger import logging


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def _get_train_transforms(self) -> transforms.Compose:
        try:
            logging.info("Creating training transforms")

            return transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.ColorJitter(
                        **self.data_transformation_config.color_jitter_transforms
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(
                        self.data_transformation_config.RANDOMROTATION
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )

        except Exception as e:
            raise XRayException(e, sys)

    def _get_test_transforms(self) -> transforms.Compose:
        try:
            logging.info("Creating testing transforms")

            return transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )

        except Exception as e:
            raise XRayException(e, sys)

    def _get_dataloaders(
        self,
        train_transform: transforms.Compose,
        test_transform: transforms.Compose,
    ) -> Tuple[DataLoader, DataLoader]:
        try:
            logging.info("Creating ImageFolder datasets and DataLoaders")

            train_dataset = ImageFolder(
                root=self.data_ingestion_artifact.train_data_path,
                transform=train_transform,
            )

            test_dataset = ImageFolder(
                root=self.data_ingestion_artifact.test_data_path,
                transform=test_transform,
            )

            train_loader = DataLoader(
                train_dataset, **self.data_transformation_config.data_loader_params
            )

            test_loader = DataLoader(
                test_dataset, **self.data_transformation_config.data_loader_params
            )

            return train_loader, test_loader

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation stage")

            train_transform = self._get_train_transforms()
            test_transform = self._get_test_transforms()

            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)

            joblib.dump(
                train_transform,
                self.data_transformation_config.train_transforms_file,
            )

            joblib.dump(
                test_transform,
                self.data_transformation_config.test_transforms_file,
            )

            train_loader, test_loader = self._get_dataloaders(
                train_transform=train_transform,
                test_transform=test_transform,
            )

            data_transformation_artifact = DataTransformationArtifact(
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                train_transform_file_path=self.data_transformation_config.train_transforms_file,
                test_transform_file_path=self.data_transformation_config.test_transforms_file,
            )

            logging.info("Data transformation completed successfully")

            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)
