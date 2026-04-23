import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def prepare_data(csv_path="data/raw/creditcard.csv"):

    data = pd.read_csv(csv_path)

    # Scale Amount
    scaler = StandardScaler()
    data['Scaled_Amount'] = scaler.fit_transform(
        data['Amount'].values.reshape(-1, 1)
    )
    data = data.drop(['Amount'], axis=1)

    # Features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # SMOTE only on training
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test