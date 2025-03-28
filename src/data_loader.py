
from src import config
from sklearn.datasets import load_iris


def load_data(config=config):
    data = load_iris(as_frame=True)
    X = data.frame.drop('target', axis=1)
    y = data.frame['target']
    X.to_csv(config.data_path, index=False)
    y.to_csv(config.target_path, index=False)


if __name__ == "__main__":
    load_data(config)