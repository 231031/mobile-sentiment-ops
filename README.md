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

- run ```docker-compose up -d``` to start airflow, mlflow, minio

## Access UI

- <http://localhost:9000> - minio ```username : minio, password : minio```
- <http://localhost:8000> - mlflow
- <http://localhost:8080> - airflow ```username : airflow, password : airflow```
