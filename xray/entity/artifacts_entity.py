from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Optional


# =========================
# Data Ingestion Artifact
# =========================
@dataclass
class DataIngestionArtifact:
    train_data_path: str
    test_data_path: str


# =========================
# Data Transformation Artifact
# =========================
@dataclass
class DataTransformationArtifact:
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    train_transform_file_path: str
    test_transform_file_path: str


# =========================
# Model Trainer Artifact
# =========================
@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    best_accuracy: float


# =========================
# Model Evaluation Artifact
# =========================
@dataclass
class ModelEvaluationArtifact:
    accuracy: float
    loss: Optional[float] = None


# =========================
# Model Pusher Artifact
# =========================
@dataclass
class ModelPusherArtifact:
    bentoml_model_name: str
    bentoml_service_name: str
