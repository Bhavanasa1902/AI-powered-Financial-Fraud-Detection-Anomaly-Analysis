import pandas as pd
import numpy as np

class featureEngineering:
    def __init__(self, time_col='timestamp', user_col='user_id'):
        self.time_col = time_col
        self.user_col = user_col
    
    # Calculate distance between user's base location and merchant's location
    def havershineDistance(self, lat1, lon1, lat2, lon2):
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # Haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        return c * r

    # Feature 1: Velocity - Transaction Frequency
    def velocityFeature(self, df):
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df = df.sort_values(by=[self.user_col, self.time_col]).reset_index(drop=True)
        
        df_time_indexed = df.set_index(self.time_col)
        
        # Transactions and amount in last 12 hours per user
        grouped = df_time_indexed.groupby(self.user_col)
        
        df['tx_count_12h'] = grouped['amount'].transform(
            lambda x: x.rolling('12h', min_periods=1).count()
        ).values
        
        df['amount_sum_12h'] = grouped['amount'].transform(
            lambda x: x.rolling('12h', min_periods=1).sum()
        ).values
        
        # Time since last transaction
        df['time_since_last_tx'] = df.groupby(self.user_col)[self.time_col].diff().dt.total_seconds().fillna(0).values
        
        return df
    
    # Feature 2: Spatial - Distance from user's base location to the merchant's location
    def spatialFeature(self, df):

        df['geo_distance_km'] = self.havershineDistance(
            df['user_base_lat'], df['user_base_lon'], 
            df['location_lat'], df['location_lon']
        )
        return df
        
    # Feature 3: Behavioral - Deviations from the user's historical spend baseline
    def behavioralFeature(self, df):
        user_means = df.groupby(self.user_col)['amount'].transform('mean')
        user_std = df.groupby(self.user_col)['amount'].transform('std').fillna(1.0)
        
        df['spend_deviation_zscore'] = (df['amount'] - user_means) / user_std
        return df

    def fit_transform(self, df):
        print("Adding Velocity features...")
        df = self.velocityFeature(df)
        print("Adding Spatial features...")
        df = self.spatialFeature(df)
        print("Adding Behavioral features...")
        df = self.behavioralFeature(df)
        return df
