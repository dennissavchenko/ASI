from app import predict, probabilities

def test_predict_returns_valid_class_id():
    sample = [5.1, 3.5, 1.4, 0.2]
    class_id = predict(sample)

    # Check type
    assert isinstance(class_id, int), "predict() should return an int"

    # Check range (Iris dataset has 3 classes: 0,1,2)
    assert 0 <= class_id <= 2, "class ID should be between 0 and 2"

def test_predict_proba_shape():
    sample = [5.1, 3.5, 1.4, 0.2]
    proba = probabilities(sample)

    # Must return shape (1,3)
    assert proba.shape == (1, 3), "predict_proba() must output probabilities for 3 classes"

    # Probabilities must sum to 1
    assert abs(proba[0].sum() - 1) < 1e-6, "probabilities must sum to 1"