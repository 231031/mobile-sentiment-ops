# MobileSentimentOps  

## 🎯 Objective

This project implements a **production-ready MLOps pipeline** for sentiment classification on mobile application reviews.  
Beyond developing a machine learning model, the focus of this work is the **full end-to-end lifecycle of ML in real-world deployment**, including:

- Reproducible experimentation and tracking (MLflow)
- Automated training, testing, and deployment workflows (CI/CD/CT)
- Model versioning, registry, and promotion
- Workflow orchestration and scheduled retraining
- Continuous monitoring of model performance and data drift
- Scalable, modular pipeline design for real production environments

This project is created as part of the subject **CPE393 - MLOps**, demonstrating how sentiment analysis models can be developed, deployed, and maintained using modern machine learning operations practices.

## Initial Project

- run ```docker-compose -f docker-compose.test.yml up -d``` to start mlflow, gcs-emulator

## Access UI

- <http://localhost:8000> - mlflow
- <http://localhost:8081> - label studio
- <http://localhost:5001> - Fast API with /predict and /retrain

## Project Dir

.
├── .github/
├── app/
├── dags/
├── data/
├── db_init/
└── lib/

### `app/`  *(FastAPI model serving + drift monitoring + retraining trigger)*

This folder contains the **production serving application**.  
It loads the Production model from MLflow, serves prediction requests, checks drift, and can trigger retraining.

- `__init__.py`  
  Marks `app/` as a Python package so modules can be imported cleanly.

- `config.py`  
  Central configuration file.  
  Stores constants and environment-based settings such as:
  - MLflow tracking URI
  - bucket/service endpoints (GCS/MinIO)
  - column names (`REVIEW_COLUMN`, `TARGET_COLUMN`)
  - alias name (`Production`)
  - report paths, Label Studio URL, etc.

- `data_pipeline.py`  
  **DataHandler**  
  Responsibilities:
  - connect to storage
  - upload any related file
  - (future) EDA / visualization helpers

- `prediction.py`  
  **PredictionHandler**  
  Responsibilities:
  - Provide model for prediction
  - run Evidently drift checks and store

- `train_model.py`  
  **MLOpsHandler / retrain handler**  
  Responsibilities:
  - train LR / RF / XGB pipelines experinement from lib folder
  - provide functions like `train_startup_model()` or function for decision to retrain and retrain

- `ml_server.py`  
  Main FastAPI server entrypoint.  
  Responsibilities:
  - lifespan startup: load Production model (or train if missing)
  - `/predict` endpoint: infer + drift check + upload outputs
  - `/retrain` endpoint: trigger retraining (usually background task)

### `lib/`

Helper scripts / utilities

- `train_with_dags.py`  
  **Traning Logic and Experinement**  
  Traning model logic and experinement in mlflow

### `db_init/`

Database initialization scripts (mostly for Label Studio / Postgres).

- `init.sh`  
  Runs when the DB container starts to create schema, users, or default tables.

---
