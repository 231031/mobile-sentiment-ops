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

for development in frontend service use ```dockerfile: Dockerfile.dev``

- run ```docker-compose -f docker-compose.test.yml up -d``` to start mlflow, gcs-emulator
- run ```docker-compose -f docker-compose.test.yml down``` down all services
- run ```docker-compose -f docker-compose.test.yml down -v``` down all services and delete all backup

## Access UI

- <http://localhost:5173> - frontend
- <http://localhost:8000> - mlflow
- <http://localhost:8080> - airflow username: airflow, password: airflow
- <http://localhost:5001> - Fast API

## Project Dir

### `app/`  *(FastAPI model serving + drift monitoring + retraining trigger)*

This folder contains the **production serving application**.  
It loads the Production model from MLflow, serves prediction requests, checks drift, and can trigger retraining.

- `data_pipeline.py`  
  **DataHandler**  
  Responsibilities:
  - connect to storage
  - upload any related file

- `prediction.py`  
  **PredictionHandler**  
  Responsibilities:
  - Provide model for prediction
  - run Evidently drift checks and store

- `ml_server.py`  
  Main FastAPI server entrypoint.  
  Responsibilities:
  - lifespan startup: load Production model (or train if missing)
  - `/predict` endpoint: handle preidction for many rows of text
  - `/predict_json` endpoint: handle prediction from text
  - `/loadmodel` endpoint: trigger to load model from airflow

### `airflow/`  *(Airflow schedule task - experinement and retrain model)*

This folder contains the **daily schedule task**.  

---

## DAGS
<img width="1246" height="323" alt="dags_pipeline" src="https://github.com/user-attachments/assets/ca237291-2091-42a1-aa3b-36bda26ef813" />
