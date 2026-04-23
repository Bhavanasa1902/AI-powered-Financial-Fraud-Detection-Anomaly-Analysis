import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw(path: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # drop duplicates if any
    df = df.drop_duplicates()
    return df

def train_val_test_split(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    val_relative = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_relative,
        stratify=y_train_full,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
