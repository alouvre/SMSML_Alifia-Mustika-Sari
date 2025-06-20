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
mlflow.autolog(disable=True)  # Nonaktifkan autolog agar tidak bentrok saat log manual


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
                "max_depth": [3, 5, 7],
                "learning_rate": [0.3, 0.1, 0.01],
                "n_estimators": [100, 200],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "gamma": [0, 1],
                "reg_lambda": [1, 10],      # L2 regularization
                "reg_alpha": [0, 1],        # L1 regularization
            }
        )
    }

    input_example = X_train[0:5]

    with mlflow.start_run(run_name="Model_Tunning") as run:
        for model_name, (model, param_grid) in models_with_params.items():
            print(f"🔍 Tuning model: {model_name}...")

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model)
            ])

            param_grid_prefixed = {f"clf__{k}": v for k, v in param_grid.items()}
            grid = GridSearchCV(pipeline, param_grid=param_grid_prefixed,
                                cv=3, scoring='accuracy',
                                n_jobs=1, error_score='raise')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)

            # 📊 Logging metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)
            false_positive_rate = fp / (fp + tn)
            false_negative_rate = fn / (fn + tp)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                f"{model_name}_accuracy": acc,
                f"{model_name}_precision": prec,
                f"{model_name}_recall": rec,
                f"{model_name}_f1_score": f1,
                f"{model_name}_specificity": specificity,
                f"{model_name}_false_positive_rate": false_positive_rate,
                f"{model_name}_false_negative_rate": false_negative_rate,
            })

            # 📁 Buat folder artifacts manual
            run_id = run.info.run_id
            artifact_dir = Path("mlartifacts") / run_id / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Simpan classification report
            report_path = artifact_dir / f"{model_name}_classification_report.json"
            log_classification_report(y_test, y_pred, report_path)

            # Simpan estimator HTML
            html_path = artifact_dir / f"{model_name}_estimator.html"
            log_estimator_html(best_model, html_path)

            # 🔁 Simpan model manual (.pkl)
            pkl_path = artifact_dir / f"{model_name}_model.pkl"
            joblib.dump(best_model, pkl_path)
            mlflow.log_artifact(str(pkl_path))

            # 🔁 Simpan model dengan struktur MLflow lengkap
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path=f"best_{model_name}_model",  # ini akan muncul sebagai folder di artifacts
                signature=signature,
                input_example=input_example
            )

            print(f"✅ {model_name} selesai dan dicatat ke MLflow.")
            print(f"🔍 Akurasi: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}  | F1-Score: {f1:.4f}")
            print(f"📈 Specificity: {specificity:.4f} | FPR: {false_positive_rate:.4f} | FPR: {false_negative_rate:.4f}")

    print("🎉 Semua model berhasil dituning dan dicatat ke MLflow.")


# Jalankan hanya jika langsung dieksekusi
if __name__ == "__main__":
    main()
