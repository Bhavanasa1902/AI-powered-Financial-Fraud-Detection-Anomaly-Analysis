"""
Fraud Detection Pipeline Orchestrator
"""
import os
import sys
# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.data.generator import generate_seeded_synthetic_dataset
from src.data.preprocessing import FraudDataPreprocessor
from src.features.engineering import featureEngineering
from src.models.supervised import SupervisedFraudModel
from src.models.unsupervised import UnsupervisedAnomalyDetector
from src.models.deep_learning import DeepAnomalyDetector
from src.models.ensemble import EnsembleScoringEngine
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.explainability import ExplainabilityModule
from src.evaluation.time_based import TimeBasedEvaluator

def run_pipeline():
    print("=== Starting AI Fraud Detection Pipeline ===\\n")
    
    # 1. Data Generation
    data_path = 'seeded_fraud_data.csv'
    if not os.path.exists(data_path):
        print("Generating Contextual Seeded Data...")
        df = pd.read_csv("C:/Users/bhava/Documents/GitHub/Credit-Card-Fraud-Detection/seeded_fraud_data.csv")
    else:
        print("Loading existing data...")
        df = pd.read_csv(data_path)
    
    # 2. Feature Engineering
    engineer = featureEngineering()
    df = engineer.fit_transform(df)
    
    # 3. Preprocessing
    preprocessor = FraudDataPreprocessor()
    df = preprocessor.load_data(data_path)
    df = preprocessor.extract_time_features(df)
    
    # Merge engineered features
    engineered_cols = ['tx_count_12h', 'amount_sum_12h', 'time_since_last_tx', 'geo_distance_km', 'spend_deviation_zscore']
    
    df_engineered = engineer.fit_transform(df.copy())
    for col in engineered_cols:
        df[col] = df_engineered[col]
        
    df.fillna(0, inplace=True)
    
    cat_cols = ['merchant_category']
    num_cols = ['amount', 'hour', 'day_of_week', 'is_weekend'] + engineered_cols
    
    df = preprocessor.encode_categorical(df, cat_cols)
    
    train_df, test_df = preprocessor.split_chronological(df, test_size=0.2)
    
    # Scale
    train_df = preprocessor.scale_numerical(train_df, num_cols, is_train=True)
    test_df = preprocessor.scale_numerical(test_df, num_cols, is_train=False)
    
    features = cat_cols + num_cols
    target = 'is_fraud'
    
    X_train_raw = train_df[features].values
    y_train_raw = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values
    
    # SMOTE
    print("\\nApplying SMOTE...")
    X_train, y_train = preprocessor.apply_smote(X_train_raw, y_train_raw)
    
    # 4. Model Training
    print("\\n--- Training Supervised Model (Random Forest) ---")
    supervised_model = SupervisedFraudModel(model_type='random_forest')
    supervised_model.train(X_train, y_train)
    sup_probs = supervised_model.predict_proba(X_test)
    
    print("\\n--- Training Unsupervised Model (Isolation Forest) ---")
    # IF trained mostly on normal
    X_train_normal = train_df[train_df[target] == 0][features].values
    unsupervised_model = UnsupervisedAnomalyDetector()
    unsupervised_model.train(X_train_normal)
    iso_scores = unsupervised_model.predict_anomaly_score(X_test)
    
    print("\\n--- Training Deep Learning Autoencoder ---")
    ae_model = DeepAnomalyDetector(input_dim=len(features), epochs=10)
    ae_model.train(X_train_normal)
    ae_scores = ae_model.predict_anomaly_score(X_test)
    
    # 5. Ensemble Scoring
    print("\\n=== Ensemble Evaluation ===")
    ensemble = EnsembleScoringEngine()
    risk_scores = ensemble.calculate_risk_score(sup_probs, iso_scores, ae_scores)
    
    # Convert scores > 70 to fraud predictions for evaluation
    y_pred_ensemble = (risk_scores > 70).astype(int)
    metrics = EvaluationMetrics.calculate_metrics(y_test, y_pred_ensemble, risk_scores/100.0)
    EvaluationMetrics.print_metrics(metrics)
    
    os.makedirs('plots', exist_ok=True)
    EvaluationMetrics.plot_confusion_matrix(y_test, y_pred_ensemble, save_path='plots/ensemble_cm.png')
    
    # 6. Explainability (SHAP)
    print("\\n=== Generating SHAP Explanations ===")
    explainer = ExplainabilityModule(supervised_model.model)
    X_test_df = pd.DataFrame(X_test, columns=features)
    
    # Local explanation for the highest risk transaction
    highest_risk_idx = np.argmax(risk_scores)
    print(f"Highest Risk Score: {risk_scores[highest_risk_idx]} for transaction {highest_risk_idx}")
    explainer.explain_instance(X_test_df.iloc[[highest_risk_idx]], save_path='plots/shap_waterfall.png')
    # explainer.plot_summary(X_test_df.sample(100), save_path='plots/shap_summary.png')
    
    # 7. Save Models and State for Real-Time API
    print("\\n=== Saving Models to Disk ===")
    os.makedirs('saved_models', exist_ok=True)
    preprocessor.save_preprocessors('saved_models/')
    supervised_model.save('saved_models/supervised_rf.pkl')
    unsupervised_model.save('saved_models/isolation_forest.pkl')
    ae_model.save('saved_models/autoencoder.pt')
    
    # Save the base dataset as a "database" to compute historical features (velocity, behavioral baseline) for new API transactions
    df.to_csv('saved_models/reference_db.csv', index=False)
    
    print("\\nPipeline execution and export complete! Ready for real-time inference.")

if __name__ == "__main__":
    run_pipeline()
