from sklearn.ensemble import IsolationForest
import numpy as np
import joblib

class UnsupervisedAnomalyDetector:
    def __init__(self, contamination=0.01, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    def build_model(self):
        # Isolation Forest maps anomalies well for multidimensional float data
        self.model = IsolationForest(
            n_estimators=150, 
            max_samples='auto', 
            contamination=self.contamination, 
            random_state=self.random_state,
            n_jobs=-1
        )

    def train(self, X_train):
        print(f"Training Isolation Forest on {X_train.shape[0]} normal transactions for baseline...")
        if self.model is None:
            self.build_model()
            
        # For an unsupervised approach, we ideally train on primarily normal data, 
        # but Isolation Forest is robust to small amounts of fraud in the training set
        self.model.fit(X_train)

    def predict_anomaly_score(self, X):
        """
        Returns anomaly scores normalized to [0, 1] where 0 = normal, 1 = anomalous.
        
        Uses decision_function() which returns:
        - Normal data: positive values (typically 0.10-0.20)
        - Anomalous data: values near 0 or negative
        
        We invert and normalize so higher = more anomalous.
        """
        scores = self.model.decision_function(X)
        # Empirical calibration:
        # Baseline ≈ 0.20 (should map to anomaly score ~0)
        # Anomaly threshold ≈ 0.0 (should map to anomaly score ~1)
        # We invert: anomaly_score = (baseline - score) / baseline
        baseline = 0.20
        inverted = (baseline - scores) / baseline
        return np.clip(inverted, 0, 1)

    def predict(self, X):
        """
        Returns 1 for anomaly (fraud), 0 for normal.
        Isolation forest returns -1 for anomaly, 1 for normal.
        """
        preds = self.model.predict(X)
        return (preds == -1).astype(int)

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"Isolation Forest Model saved to {path}")

    def load(self, path):
        self.model = joblib.load(path)
