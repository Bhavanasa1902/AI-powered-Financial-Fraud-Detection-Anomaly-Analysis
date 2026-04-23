import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

class FraudDataPreprocessor:
    def __init__(self, target_col='is_fraud', time_col='timestamp'):
        self.target_col = target_col
        self.time_col = time_col
        self.scaler = RobustScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        # Load dataset and sort chronologically
        df = pd.read_csv(file_path)
        if self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(df[self.time_col])
            df = df.sort_values(self.time_col).reset_index(drop=True)
        return df

    def extract_time_features(self, df):
        if self.time_col in df.columns:
            df['hour'] = df[self.time_col].dt.hour
            df['day_of_week'] = df[self.time_col].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df

    def encode_categorical(self, df, categorical_cols):
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen labels in test set
                classes = self.label_encoders[col].classes_
                df[col] = df[col].apply(lambda x: self.label_encoders[col].transform([x])[0] if x in classes else -1)
        return df

    def scale_numerical(self, df, numerical_cols, is_train=True):
        if is_train:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        return df

    def split_chronological(self, df, test_size=0.2):
        """
        Splits data chronologically to mimic real-world fraud detection scenarios 
        (predicting future fraud based on past data).
        """
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test
        
    def apply_smote(self, X_train, y_train):
        """
        Applies SMOTE to handle the extreme class imbalance in training data.
        """
        smote = SMOTE(sampling_strategy='minority', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

    def save_preprocessors(self, path_prefix=''):
        joblib.dump(self.scaler, f'{path_prefix}scaler.pkl')
        joblib.dump(self.label_encoders, f'{path_prefix}label_encoders.pkl')

    def load_preprocessors(self, path_prefix=''):
        self.scaler = joblib.load(f'{path_prefix}scaler.pkl')
        self.label_encoders = joblib.load(f'{path_prefix}label_encoders.pkl')
