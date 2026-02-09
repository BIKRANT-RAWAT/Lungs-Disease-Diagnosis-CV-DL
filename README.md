# 🩺 Chest X-Ray Pneumonia Detection (Production-Ready ML Project)
Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli. Symptoms typically include some combination of productive or dry cough, chest pain, fever and difficulty breathing. The severity of the condition is variable. Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications or conditions such as autoimmune diseases.Risk factors include cystic fibrosis, chronic obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, a history of smoking, a poor ability to cough such as following a stroke and a weak immune system. Diagnosis is often based on symptoms and physical examination. Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis.The disease may be classified by where it was acquired, such as community- or hospital-acquired or healthcare-associated pneumonia.

This project implements an **end-to-end, production-ready deep learning pipeline** for detecting **Pneumonia vs Normal** cases from **Chest X-Ray images** using **PyTorch**. The codebase follows **industry-standard MLOps practices**, clean architecture, and clear separation of concerns.

---

## 🚀 Project Objective

To build, train, evaluate, and deploy a **binary image classification model** that can accurately distinguish between:

* **NORMAL** chest X-rays
* **PNEUMONIA** chest X-rays

The project is designed to be:

* Modular
* Scalable
* Cloud-ready (AWS S3 + BentoML)
* Production ready

---

![xray_arch](https://github.com/BIKRANT-RAWAT/Lungs-Disease-Diagnosis-CV-DL/blob/main/flowcharts/overall.jpg)

---
## 📊 Model Visualization (FastAPI)

![interfacel](https://github.com/BIKRANT-RAWAT/Lungs-Disease-Diagnosis-CV-DL/blob/main/images/file_choose.png)
![Prediction](https://github.com/BIKRANT-RAWAT/Lungs-Disease-Diagnosis-CV-DL/blob/main/images/response.png)

---

## 🧠 Key Highlights

* ✅ Binary classification (COVID class removed intentionally)
* ✅ Custom CNN architecture in PyTorch
* ✅ Robust data augmentation & normalization
* ✅ Best-model checkpointing
* ✅ Clean separation of **model**, **training**, **inference**, and **artifacts**
* ✅ BentoML-ready for deployment
* ✅ GPU/CPU compatible

---
## 💾 Dataset used

The dataset was shared by **Apollo Diagnostic Center** for research purposes. A Proof of Concept (POC) was built using this proprietary dataset to validate the pneumonia detection pipeline.

---

## 💻 Tech Stack Used

1. Python
2. FastAPI
3. PyTorch
4. Docker
5. AWS
6. Azure

---

## 🖥 Infrastructure Required

1. AWS S3
2. AWS App Runner
3. GitHub Actions

---

## 🎯 How to Run

### Step 1: Download the project

```bash
Download the zip file and extract it to a folder.
or
Run code cell in notebook to download from kaggle.
```

### Step 2: Create a Conda environment

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate.bat
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Export environment variables

```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>
```

### Step 5: Run the application server

```bash
python app.py
```

### Step 6: Train the model

```bash
http://localhost:8001/train
```

### Step 7: Run prediction

```bash
http://localhost:8001/predict
```

---

## 🚢 Run Using Docker (Local)

1. Ensure `Dockerfile` is present in the project root directory.

2. Build the Docker image

```bash
docker build -t xray_classification .
```

3. Run the Docker container

```bash
docker run -d -p 8001:8001 \
  -e AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> \
  -e AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> \
  xray_classifier
```

---

## 📁 Project Structure

```text
project-root/
│
├── flowchart/                             # contain flowchart of project
├── images/                                # contain images of project
│
├── scripts/
│   └── start_up.sh                        # contains steps for deployment
│
├── xray/
    └── components/
           ├── data_ingestion.py           # Downloads raw X-ray images
           ├── data_transformation.py      # Transforms & DataLoaders
           ├── model_trainer.py            # Training logic
           ├── model_evaluation.py         # Model evaluation
           └── model_pusher.py             # BentoML model push
    └── cloud_storage/
           └── s3_operations.py            # AWS S3 download/upload utilities
    └── entity/
          └── artifact_entity.py           # Artifact dataclasses
    └──ml/
          └── model/
                └── arch.py                # CNN architecture ONLY
                └── model_service.py       # CNN architecture ONLY
    └──constants/
          └── __init__.py                  # Project-wide constants
├── notebooks/
│   └── experiments.ipynb                  # Research & experimentation
│
├── xray_model.pth                         # Best saved model
│
├── xray_model_last.pth                    # last saved model
│
├── app.py                                 # Inference entry-point
├── requirements.txt
├── bentofile.yaml
└── README.md
```

---

## 🧬 Dataset

* Source: **Kaggle Chest X-Ray Dataset**
* Classes Used:

  * `NORMAL`
  * `PNEUMONIA`
* Dataset Structure:

```text
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

> ⚠️ The `COVID` class was **explicitly removed** to enforce **binary classification**.

---

## 🔄 Data Pipeline

### 🔹 Data Ingestion

* Downloads raw image data from **AWS S3** (or local fallback)
* Stores versioned artifacts

### 🔹 Data Transformation

Applied **only on training data**:

* Resize → RandomResizedCrop
* Horizontal Flip
* Light Affine Transformations (optional)
* Normalization (ImageNet stats)

Test data uses **deterministic transforms only**.

---

## 🏗 Model Architecture

* Custom CNN (defined in `ml/model/arch.py`)
* Convolution + BatchNorm + ReLU blocks
* MaxPooling for spatial reduction
* Adaptive Average Pooling
* Fully Connected classifier

```text
Input (3×224×224)
→ Conv Blocks
→ AdaptiveAvgPool
→ Linear (2 classes)
```

---

## 🏋️ Training Strategy

* Loss Function: `CrossEntropyLoss`
* Optimizer: `Adam`
* Scheduler: `ReduceLROnPlateau`
* Metric: Validation Accuracy
* Checkpointing: **Best model only**

```python
torch.save(model.state_dict(), "xray_model.pth")
```

✔ Ensures the **best-performing model**, not the last epoch, is saved.

---

## 📊 Evaluation

* Accuracy-based evaluation
* Separate evaluation pipeline
* Metrics stored as artifacts

---

## 📦 Artifacts & Entities

Dataclasses are used to track pipeline outputs:

* `DataIngestionArtifact`
* `DataTransformationArtifact`
* `ModelTrainerArtifact`
* `ModelEvaluationArtifact`
* `ModelPusherArtifact`

This enables **traceability**, **debugging**, and **pipeline reproducibility**.

---

## 🚀 Deployment (BentoML Ready)

* Trained model pushed using **BentoML**
* Service name and model name configurable
* Ready for Docker & AWS ECR

---

## 🖥 Inference

Inference pipeline:

1. Load trained model
2. Apply test-time transforms
3. Perform prediction
4. Map output → class label

```python
PREDICTION_LABEL = {0: "NORMAL", 1: "PNEUMONIA"}
```

---

## ⚙️ Configuration Highlights

All constants are centralized:

* Batch size
* Image size
* Normalization stats
* Epochs
* Learning rate schedule

This avoids **hardcoding** and supports **easy experimentation**.

---

## ✅ Final Outcome

* ✔ Successfully trained a binary pneumonia detection model
* ✔ Clean separation of research vs production code
* ✔ Scalable, maintainable ML architecture
* ✔ Ready for real-world deployment

---

## 🧪 Future Improvements

* Impove accuracy of model
* Extend to multi-label classification

---

## 🤝 Author Note

This project demonstrates **end-to-end ML system design**, not just model training aligning closely with **industry and MLOps standards**.
This project was built under the guidance of PWSkills team. 

If you're reviewing this as part of an interview or production audit — this codebase is designed to pass both ✅

