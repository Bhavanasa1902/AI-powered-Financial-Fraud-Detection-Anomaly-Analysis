import numpy as np

class EnsembleScoringEngine:
    def __init__(self, weights=None):
        """
        Ensemble to vote or average scores from multiple distinct model architectures.
        Weights are tuned for real-time inference where the Isolation Forest provides
        the most consistent anomaly signal.
        """
        if weights is None:
            self.weights = {
                'supervised': 0.30,
                'isolation_forest': 0.50,
                'autoencoder': 0.20
            }
        else:
            self.weights = weights

    def calculate_risk_score(self, supervised_proba, iforest_score, ae_score):
        """
        Inputs: 
        - supervised_proba: Array of [0, 1] probabilties from Random Forest
        - iforest_score: Array of [0, 1] anomaly scores from Isolation Forest
        - ae_score: Array of [0, 1] normalized reconstruction errors from Autoencoder
        
        Returns integer arrays 0-100 indicating risk.
        Uses a weighted average. Contextual rule-based boosting in app.py
        handles cases where obvious fraud indicators are present.
        """
        risk_score = (
            supervised_proba * self.weights['supervised'] +
            iforest_score * self.weights['isolation_forest'] +
            ae_score * self.weights['autoencoder']
        )
        
        # Scale to 0-100
        risk_score_100 = np.clip(np.round(risk_score * 100), 0, 100).astype(int)
        return risk_score_100

    def get_decision(self, risk_score):
        """
        Decision engine that maps risk score to action.
        Thresholds are calibrated for the ensemble's output range,
        accounting for baseline noise from the anomaly detectors.
        0-30: ALLOW
        31-45: STEP_UP_AUTH (SMS/App prompt)
        46-60: FLAG_FOR_REVIEW
        61-100: BLOCK
        """
        if risk_score <= 30:
            return "ALLOW"
        elif risk_score <= 45:
            return "STEP_UP_AUTH"
        elif risk_score <= 60:
            return "FLAG_FOR_REVIEW"
        else:
            return "BLOCK"
