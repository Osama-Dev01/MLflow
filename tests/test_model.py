import pytest
import numpy as np
from src.model import SimpleClassifier, create_sample_data

class TestSimpleClassifier:
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        model = SimpleClassifier()
        assert model.is_trained == False
        assert model.model is not None
    
    def test_train_model(self):
        """Test training works and returns metrics"""
        model = SimpleClassifier()
        X, y = create_sample_data()
        
        result = model.train(X, y)
        
        assert model.is_trained == True
        assert "accuracy" in result
        assert 0 <= result["accuracy"] <= 1
    
    def test_predict_after_training(self):
        """Test prediction works after training"""
        model = SimpleClassifier()
        X, y = create_sample_data()
        
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_before_training(self):
        """Test that prediction without training raises error"""
        model = SimpleClassifier()
        X, _ = create_sample_data()
        
        with pytest.raises(ValueError, match="Model not trained yet!"):
            model.predict(X)
    
    def test_model_info(self):
        """Test model metadata is correct"""
        model = SimpleClassifier()
        
        # Before training
        info = model.get_model_info()
        assert info["is_trained"] == False
        assert info["features"] is None
        
        # After training
        X, y = create_sample_data()
        model.train(X, y)
        info = model.get_model_info()
        assert info["is_trained"] == True
        assert info["features"] == 2

class TestDataCreation:
    
    def test_sample_data_shape(self):
        """Test that sample data has correct dimensions"""
        X, y = create_sample_data()
        assert X.shape == (100, 2)
        assert y.shape == (100,)
    
    def test_sample_data_types(self):
        """Test data types are correct"""
        X, y = create_sample_data()
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)