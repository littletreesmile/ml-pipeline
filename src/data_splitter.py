
import pandas as pd
from sklearn.model_selection import train_test_split
from src import config


def split_data(config, test_size=0.2, random_state=123, shuffle=True):
    X = pd.read_csv(config.data_path)
    y = pd.read_csv(config.target_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_data(config=config, test_size=0.2, random_state=123, shuffle=True)

