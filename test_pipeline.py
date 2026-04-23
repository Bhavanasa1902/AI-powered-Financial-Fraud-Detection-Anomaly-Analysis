from src.preprocessing import prepare_data

X_train, X_test, y_train, y_test = prepare_data()

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Balanced training distribution:")
print(y_train.value_counts())