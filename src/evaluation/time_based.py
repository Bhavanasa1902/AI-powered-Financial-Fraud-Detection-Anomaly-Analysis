import pandas as pd
from datetime import timedelta
from src.evaluation.metrics import EvaluationMetrics

class TimeBasedEvaluator:
    def __init__(self, model_class, time_col='timestamp'):
        self.model_class = model_class
        self.time_col = time_col

    def simulate_degradation(self, df, features, target='is_fraud', train_months=3, test_months=1):
        """
        Simulates model performance degradation over time on distinct test blocks
        without retraining.
        """
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        start_date = df[self.time_col].min()
        train_end = start_date + pd.DateOffset(months=train_months)
        
        train_df = df[df[self.time_col] < train_end]
        X_train, y_train = train_df[features], train_df[target]
        
        model = self.model_class()
        model.train(X_train.values, y_train.values)
        
        results = []
        current_date = train_end
        
        print("Evaluating over time...")
        for i in range(1, 4): # Test for the next 3 months
            period_end = current_date + pd.DateOffset(months=test_months)
            test_df = df[(df[self.time_col] >= current_date) & (df[self.time_col] < period_end)]
            
            if len(test_df) == 0:
                break
                
            X_test, y_test = test_df[features], test_df[target]
            y_prob = model.predict_proba(X_test.values)
            y_pred = model.predict(X_test.values)
            
            metrics = EvaluationMetrics.calculate_metrics(y_test, y_pred, y_prob)
            metrics['Period'] = f"Month {train_months + i}"
            metrics['Num_Transactions'] = len(test_df)
            results.append(metrics)
            
            current_date = period_end
            
        return pd.DataFrame(results)
