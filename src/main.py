import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models import infer_signature

from src import config
from data_loader import load_data
from data_splitter import split_data
from pipeline import create_pipeline


params = {
    "feature_range": (0, 1),
    "max_iter": 100
}


def train_pipeline(params):
    load_data()
    X_train, X_test, y_train, y_test = split_data(config, test_size=0.2, random_state=123, shuffle=True)
    pipeline = create_pipeline(params["feature_range"], params["max_iter"])
    pipeline.fit(X_train, y_train.values.ravel())
    score = pipeline.score(X_test, y_test.values.ravel())
    print(score)


    with mlflow.start_run(run_name="ml-pipeline-run"):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", score)
        signature = infer_signature(X_train, pipeline.predict(X_train))
        model_info = mlflow.sklearn.log_model(sk_model=pipeline,
                                 artifact_path="modelss",
                                 signature=signature,
                                 input_example=X_train)
    return model_info


if __name__ == "__main__":
    params = {
        "feature_range": (0, 1),
        "max_iter": 20
    }

    mlflow.set_tracking_uri(uri=config.mlflow_tracking_uri)
    mlflow.set_experiment("ML Pipeline")
    model_info = train_pipeline(params)

    # Prediction
    X = pd.read_csv(config.data_path)
    y = pd.read_csv(config.target_path)
    sklearn_pyfunc = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    predictions = sklearn_pyfunc.predict(X)
    accuracy = accuracy_score(predictions, y)
    print(f"Prediction accuracy: {accuracy}")