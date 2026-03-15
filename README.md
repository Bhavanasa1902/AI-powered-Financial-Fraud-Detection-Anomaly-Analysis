# AI-Powered Fraud Detection System

## Overview
This repository contains a full end-to-end framework for analyzing and classifying financial transactions using a multi-model approach designed explicitly for highly imbalanced fraud scenarios. The system emphasizes anomaly detection, complex feature engineering, and Explainable AI (SHAP) to assist fraud analysts, effectively lowering false positive rates and increasing the transparency of flagged transactions.

## Setup and Installation
The code is written in Python. Note that due to modern environment constraints, it is highly recommended to run this inside a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline
You can run the full pipeline cleanly using the provided script in the `notebooks` directory.

```bash
python notebooks/run_pipeline.py
```

This script will sequentially:
1. **Generate Data**: Build a highly-realistic, synthetic structured dataset featuring over 20,000 transactions mimicking actual extreme Class Imbalance patterns (0.5% fraud) and localized behaviors. 
2. **Feature Engineer**: Inject chronological properties like time-since-last-transaction, rolling velocity spend windows, haversine geographical displacement, and behavioral anomalies. 
3. **Train Supervised Models**: Fit an XGBoost classifier with heavily weighted minority classes (`scale_pos_weight`).
4. **Train Unsupervised Models**: Compute baseline expected behaviors through a trained `Isolation Forest` and identify structural deviations using a PyTorch autoencoder.
5. **Ensemble Scoring**: Weighted ensemble to map model inferences into a direct 0-100 risk score matrix. 
6. **Evaluate & Explain**: SHAP tree explainers isolate exactly *why* a transaction was blocked.

## Architecture

* **`src/data/`**: Data mocking and chronological SMOTE over-sampling.
* **`src/features/`**: Spatial distance tracking, behavioral delta modeling.
* **`src/models/`**: Implementation of Machine Learning architectures (`XGBoost`, `Random Forest`, `Isolation Forest`, `LSTM/Autoencoder`).
* **`src/evaluation/`**: `shap` explainability implementations and metric trackers for F1, PR-AUC, and ROC curves.

## Dashboards
Check the locally generated `plots` directory after running the orchestrator to view generated confusion matrices, and SHAP feature importance charts.

## Live Simulation UI
A real-time Flask API and checkout simulator was built to demonstrate how this system ingests live transaction streams.
To run the web interface:

```bash
source venv/bin/activate
python src/api/app.py
```
Then navigate to `http://localhost:5001/` in your browser.
