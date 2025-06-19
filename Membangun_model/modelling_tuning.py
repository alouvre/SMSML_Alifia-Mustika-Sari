import json
from pathlib import Path

import mlflow
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from mlflow.models.signature import infer_signature

# ----------------------------------
# 🔧 Setup MLflow tracking lokal
# ----------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Dropout_Prediction_Submission")


# ----------------------------------
# 📂 Load dan split data
# ----------------------------------
def load_data(path="data_student_cleaned.csv"):
    df = pd.read_csv(path)
    X = df.drop("Status", axis=1)
    y = df["Status"]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


#  Simpan classification report
def log_classification_report(y_true, y_pred, filename="classification_report.json"):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(filename, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(filename)


# Simpan confusion matrix
def log_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(filename)
        plt.close()
        mlflow.log_artifact(filename)
    else:
        print(f"⚠️ Confusion matrix shape {cm.shape} is not 2x2. Skipped logging.")


def log_estimator_html(model, filename="estimator.html"):
    with open(filename, "w") as f:
        f.write("<html><body><h2>Best Estimator</h2><pre>")
        f.write(str(model))
        f.write("</pre></body></html>")
    mlflow.log_artifact(filename)


def main():
    X_train, X_test, y_train, y_test = load_data()

    models_with_params = {
        "XGBoost": (
            XGBClassifier(eval_metric='logloss', n_jobs=1),
            {
                "max_depth": [3, 5],
                "learning_rate": [0.1, 0.01],
                "n_estimators": [100, 200],
            }
        )
    }
    with mlflow.start_run(run_name="Model_Tunning") as run:
        for model_name, (model, param_grid) in models_with_params.items():
            print(f"🔍 Tuning model: {model_name}...")

            mlflow.autolog(disable=True)  # Nonaktifkan autolog agar tidak bentrok saat log manual

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model)
            ])

            param_grid_prefixed = {f"clf__{k}": v for k, v in param_grid.items()}
            grid = GridSearchCV(pipeline, param_grid=param_grid_prefixed, cv=3, scoring='accuracy', n_jobs=1, error_score='raise')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)

            # 📊 Logging metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                f"{model_name}_accuracy": acc,
                f"{model_name}_precision": prec,
                f"{model_name}_recall": rec,
                f"{model_name}_f1_score": f1
            })

            # 📁 Buat folder artifacts manual
            run_id = run.info.run_id
            artifact_dir = Path("mlartifacts") / run_id / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Simpan classification report
            report_path = artifact_dir / f"{model_name}_classification_report.json"
            log_classification_report(y_test, y_pred, report_path)

            # Simpan confusion matrix
            # cm_path = artifact_dir / f"{model_name}_confusion_matrix.png"
            # log_confusion_matrix(y_test, y_pred, cm_path)
            # mlflow.log_figure(cm.figure_, 'test_confusion_matrix.png')

            # Simpan estimator HTML
            html_path = artifact_dir / f"{model_name}_estimator.html"
            log_estimator_html(best_model, html_path)

            # 🔁 Simpan model manual (.pkl)
            pkl_path = artifact_dir / f"{model_name}_model.pkl"
            joblib.dump(best_model, pkl_path)
            mlflow.log_artifact(str(pkl_path))

            # 🔁 Simpan model dengan struktur MLflow lengkap
            signature = infer_signature(X_test, y_pred)
            # saved_model_dir = artifact_dir / f"best_{model_name}_model"
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path=f"{model_name}_mlflow_model",  # ini akan muncul sebagai folder di artifacts
                signature=signature
            )
            # mlflow.sklearn.save_model(best_model, path=str(saved_model_dir), signature=signature)
            # mlflow.log_artifacts(str(saved_model_dir))

            print(f"✅ {model_name} selesai dan dicatat ke MLflow.")

    print("🎉 Semua model berhasil dituning dan dicatat ke MLflow.")


# Jalankan hanya jika langsung dieksekusi
if __name__ == "__main__":
    main()
