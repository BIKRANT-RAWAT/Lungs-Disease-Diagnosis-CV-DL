from datetime import datetime
from typing import List, Dict
import torch

# =========================
# General Constants
# =========================
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACT_DIR: str = "artifacts"
TRAINED_MODEL_DIR: str = "trained_model"

# =========================
# Data / Storage
# =========================
BUCKET_NAME: str = "lungxray"
S3_DATA_FOLDER: str = "data"

# =========================
# Class Labels
# =========================
CLASS_LABEL_1: str = "NORMAL"
CLASS_LABEL_2: str = "PNEUMONIA"

CLASS_NAMES: List[str] = [CLASS_LABEL_1, CLASS_LABEL_2]

PREDICTION_LABEL: Dict[int, str] = {
    0: CLASS_LABEL_1,
    1: CLASS_LABEL_2
}

# =========================
# Image Preprocessing
# =========================
RESIZE: int = 256
CROP_SIZE: int = 224

HORIZONTAL_FLIP_PROB: float = 0.5

# Normalization (ImageNet – compatible with medical CNNs)
NORMALIZE_MEAN: List[float] = [0.485, 0.456, 0.406]
NORMALIZE_STD: List[float] = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS_KEY: str = "xray_train_transforms"
TRAIN_TRANSFORMS_FILE: str = "train_transforms.pkl"
TEST_TRANSFORMS_FILE: str = "test_transforms.pkl"

# =========================
# DataLoader Parameters
# =========================
BATCH_SIZE: int = 16
SHUFFLE_TRAIN: bool = True
SHUFFLE_TEST: bool = False
PIN_MEMORY: bool = True
NUM_WORKERS: int = 2

# =========================
# Training Configuration
# =========================
EPOCHS: int = 15
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4

# =========================
# Device
# =========================
DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================
# Model Saving
# =========================
TRAINED_MODEL_NAME: str = "xray_model.pth"

# =========================
# Deployment (BentoML)
# =========================
BENTOML_MODEL_NAME: str = "xray_model"
BENTOML_SERVICE_NAME: str = "xray_service"
BENTOML_ECR_IMAGE: str = "xray_bento_image"
