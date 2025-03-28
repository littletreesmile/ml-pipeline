from sklearn.pipeline import Pipeline
from data_preprocessing import CustomScaler
from model import create_model


def create_pipeline(feature_range, max_iter):
    scaler = CustomScaler(feature_range)
    model = create_model(max_iter)
    return Pipeline([('scaler', scaler), ('model', model)])