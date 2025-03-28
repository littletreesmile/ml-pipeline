
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score

from src import config

mlflow.set_tracking_uri(uri=config.mlflow_tracking_uri)
m = mlflow.pyfunc.load_model(f"models:/model_1/None")  #

# Prediction
X = pd.read_csv(config.data_path)
y = pd.read_csv(config.target_path)
y_pred = m.predict(X)
accuracy = accuracy_score(y_pred, y)
print(accuracy)