
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_range_ = None

    def fit(self, X, y=None):
        self.data_min_ = np.min(X, axis=0)
        self.data_range_ = np.max(X, axis=0) - self.data_min_
        self.data_range_[self.data_range_ == 0] = 1  # Avoid division by zero
        return self  # Transformer must return self according to scikit-learn convention

    def transform(self, X, y=None):
        x_std = (X - self.data_min_) / self.data_range_
        x_scaled = x_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return x_scaled