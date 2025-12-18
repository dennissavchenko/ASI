import mlflow.sklearn
import json
import joblib
from mlflow import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=42
)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=200),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}


mlflow.set_experiment("iris-model-zoo")

accuracy = -1
best_f1 = -1
best_model = None
best_run_id = None
best_model_name = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Log params
        mlflow.log_param("model_name", name)

        # Classification report
        report = classification_report(y_test, preds)
        with open(f"reports/{name}/classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact(f"reports/{name}/classification_report.txt")

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(name)
        plt.savefig(f"reports/{name}/confusion_matrix.png")
        plt.close()
        mlflow.log_artifact(f"reports/{name}/confusion_matrix.png")

        # Log model
        mlflow.sklearn.log_model(
            model,
            name=f"{name}-iris-model",
            input_example=X_train[:1]
        )

        # Track the best model
        if f1 > best_f1:
            accuracy = acc
            best_f1 = f1
            best_model = model
            best_model_name = name
            best_run_id = mlflow.active_run().info.run_id

client = MlflowClient()

model_name = "IrisModel"

# Create a registered model if it doesn't exist
try:
    client.get_registered_model(model_name)
except:
    client.create_registered_model(model_name)

# Register a new version from the BEST run
model_uri = f"runs:/{best_run_id}/model"

client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=best_run_id
)

joblib.dump(best_model, "app/model.joblib")

meta = {
    "best_model": best_model_name,
    "metrics": {
        "accuracy": accuracy,
        "f1_macro": best_f1
    },
    "mlflow_run_id": best_run_id,
    "version": "v1.1.0"
}

with open("app/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
