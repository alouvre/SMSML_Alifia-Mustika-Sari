import json
from pathlib import Path

import mlflow
import pandas as pd

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# ----------------------------------
# 🔧 Setup MLflow tracking lokal
# ----------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Dropout_Prediction_Submission_NoTuning")
mlflow.autolog()  # ✅ Aktifkan autolog


# ----------------------------------
# 📂 Load dan split data
# ----------------------------------
def load_data(path="data_student_cleaned.csv"):
    df = pd.read_csv(path)
    X = df.drop("Status", axis=1)
    y = df["Status"]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def log_classification_report(y_true, y_pred, filename):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(filename, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(filename)


def log_estimator_html(model, filename):
    with open(filename, "w") as f:
        f.write("<html><body><h2>Best Estimator</h2><pre>")
        f.write(str(model))
        f.write("</pre></body></html>")
    mlflow.log_artifact(filename)


def main():
    X_train, X_test, y_train, y_test = load_data()

    model_name = "XGBoost"
    model = XGBClassifier(eval_metric='logloss', n_jobs=1)

    with mlflow.start_run(run_name="Model_NoTuning") as run:
        print(f"🔍 Training model: {model_name}...")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metrics({
                f"test_{model_name}_accuracy_score": acc,
                f"test_{model_name}_precision_score": prec,
                f"test_{model_name}_recall_score": rec,
                f"test_{model_name}_f1_score": f1
            })

        # 📁 Buat folder artifacts manual
        run_id = run.info.run_id
        artifact_dir = Path("mlartifacts") / run_id / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Logging artifacts
        log_classification_report(y_test, y_pred, artifact_dir / f"{model_name}_metric_info.json")
        log_estimator_html(pipeline, artifact_dir / f"{model_name}_estimator.html")

        print(f"✅ Model {model_name} selesai dilatih dan dicatat ke MLflow.")
        print(f"🔍 Akurasi: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}  | F1-Score: {f1:.4f}")

    print("🎉 Proses selesai tanpa tuning.")


if __name__ == "__main__":
    main()
