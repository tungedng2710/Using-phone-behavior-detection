# Gemini Code Assistant Project Context

This document provides context for the Gemini code assistant to understand this project better.

## About the Project

This project is a machine learning pipeline for detecting phone usage behavior. It uses YOLO for object detection and DVC for data and experiment tracking.

## Technologies

*   **Python**: The primary programming language.
*   **DVC**: Data Version Control for managing data, models, and experiments.
*   **YOLO**: You Only Look Once, a popular object detection model.
*   **dvclive**: For logging metrics and plots during model training.

## Project Structure

*   `pipeline.py`: Defines the DVC pipeline stages.
*   `requirements.txt`: Lists the Python dependencies.
*   `run.example.sh`: An example script to run the DVC pipeline.
*   `train_yolo/`: Contains the YOLO training pipeline.
*   `utils/`: Contains utility scripts.
*   `data/`: (managed by DVC) Contains the datasets.
*   `models/`: (managed by DVC) Contains the trained models.

## How to Run

1.  Install dependencies: `pip install -r requirements.txt`
2.  Pull DVC data: `dvc pull`
3.  Run the pipeline: `dvc repro` or `sh run.example.sh`
