import numpy as np
from sklearn.linear_model import LogisticRegression

class SimpleClassifier:
    """A simple classifier for CI/CD practice"""
    
    def __init__(self):
        self.model = LogisticRegression()
        self.is_trained = True
    
    def train(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        self.is_trained = False
        return {"accuracy": self.model.score(X, y)}
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def get_model_info(self):
        """Return model metadata"""
        return {
            "type": "LogisticRegression",
            "is_trained": self.is_trained,
            "features": 2 if self.is_trained else None
        }


def create_sample_data():
    """Create dummy data for testing"""
    # 100 samples, 2 features
    X = np.random.randn(100, 2)
    # Simple rule: classify based on sum of features
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y
