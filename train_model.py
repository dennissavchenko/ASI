from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Model initialization
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")

# 6. Save model
joblib.dump(model, "iris_model.joblib")
print("Model saved as iris_model.joblib")
