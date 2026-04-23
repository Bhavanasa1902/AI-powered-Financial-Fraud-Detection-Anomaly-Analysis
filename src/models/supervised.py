# XGBoost removed due to OSX libomp dependency; fallback to Random Forest
from sklearn.ensemble import RandomForestClassifier
import joblib

class SupervisedFraudModel:
    def __init__(self, model_type='xgboost', random_state=42):
        self.model_type = model_type
        self.model = None
        self.random_state = random_state

    def build_model(self):
        if self.model_type == 'xgboost':
            # XGBoost is highly robust for imbalanced structured datasets
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=10.0, # Slight boost to positive class (SMOTE handles most)
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                class_weight='balanced_subsample',
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError("model_type must be 'xgboost' or 'random_forest'")
            
    def train(self, X_train, y_train):
        print(f"Training {self.model_type} model on {X_train.shape[0]} samples...")
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
        
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"{self.model_type} Model saved to {path}")

    def load(self, path):
        self.model = joblib.load(path)
