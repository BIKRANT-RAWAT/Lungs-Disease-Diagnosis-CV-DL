import os
from dataclasses import dataclass
from torch import device

from xray.constant.training_pipeline import *


# =========================
# Data Ingestion Config
# =========================
@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.bucket_name: str = BUCKET_NAME
        self.s3_data_folder: str = S3_DATA_FOLDER

        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)

        self.data_path: str = os.path.join(
            self.artifact_dir, "data_ingestion", self.s3_data_folder
        )

        self.train_data_path: str = os.path.join(self.data_path, "train")
        self.test_data_path: str = os.path.join(self.data_path, "test")


# =========================
# Data Transformation Config
# =========================
@dataclass
class DataTransformationConfig:
    def __init__(self):
        # Image sizes
        self.resize: int = RESIZE
        self.crop_size: int = CROP_SIZE

        # Augmentations (final decision)
        self.horizontal_flip_prob: float = HORIZONTAL_FLIP_PROB

        # Normalization
        self.normalize_transforms: dict = {
            "mean": NORMALIZE_MEAN,
            "std": NORMALIZE_STD,
        }

        # DataLoader params
        self.train_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": SHUFFLE_TRAIN,
            "pin_memory": PIN_MEMORY,
            "num_workers": NUM_WORKERS,
        }

        self.test_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": SHUFFLE_TEST,
            "pin_memory": PIN_MEMORY,
            "num_workers": NUM_WORKERS,
        }

        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR, TIMESTAMP, "data_transformation"
        )

        self.train_transforms_file: str = os.path.join(
            self.artifact_dir, TRAIN_TRANSFORMS_FILE
        )

        self.test_transforms_file: str = os.path.join(
            self.artifact_dir, TEST_TRANSFORMS_FILE
        )


# =========================
# Model Trainer Config
# =========================
@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR, TIMESTAMP, "model_training"
        )

        self.trained_bentoml_model_name: str = BENTOML_MODEL_NAME

        self.trained_model_path: str = os.path.join(
            self.artifact_dir, TRAINED_MODEL_NAME
        )

        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY

        # Training params
        self.epochs: int = EPOCHS
        self.learning_rate: float = LEARNING_RATE
        self.weight_decay: float = WEIGHT_DECAY

        # Optimizer (Adam)
        self.optimizer_params: dict = {
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        }

        self.device: device = DEVICE


# =========================
# Model Evaluation Config
# =========================
@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.device: device = DEVICE

        self.best_accuracy: float = 0.0
        self.best_loss: float = float("inf")

        self.total_samples: int = 0
        self.total_batches: int = 0


# =========================
# Model Pusher Config
# =========================
@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.bentoml_model_name: str = BENTOML_MODEL_NAME
        self.bentoml_service_name: str = BENTOML_SERVICE_NAME
        self.bentoml_ecr_image: str = BENTOML_ECR_IMAGE
        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY
