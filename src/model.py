from sklearn.linear_model import LogisticRegression

def create_model(max_iter):
    return LogisticRegression(max_iter=max_iter)