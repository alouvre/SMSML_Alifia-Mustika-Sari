import mlflow
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import StandardScaler

# Tracking ke lokal (atau ganti ke DagsHub jika mau)
# ----------------------------------
# 🔐 Setup Local Tracking
# ----------------------------------
# dagshub.init(
#     repo_owner='alouvre',
#     repo_name='SMSML_Alifia-Mustika-Sari',
#     mlflow=True
# )
# mlflow.set_tracking_uri("https://dagshub.com/alouvre/SMSML_Alifia-Mustika-Sari.mlflow")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Dropout_Prediction_Tuning")

# ----------------------------------
# ⚙️ Load Data and Split train-test
# ----------------------------------
data = pd.read_csv("data_student_clean.csv")

# Pisahkan fitur dan target
X = data.drop("Status", axis=1)
y = data["Status"]

# Split data: 75% training, 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y  # stratify agar proporsi kelas tetap seimbang
)

# ----------------------------------
# ⚙️ Define Model & Hyperparameter Grid
# ----------------------------------
models_with_params = {
    'XGBoost': (
        XGBClassifier(eval_metric='logloss'),
        {
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [100, 200]
        }
    )
}

# ----------------------------------
# ✅ Logging ke MLflow lokal
# ----------------------------------
with mlflow.start_run(run_name="Model_Tunning"):
    for model_name, (model, param_grid) in models_with_params.items():
        print(f"🔍 Tuning model: {model_name}...")
        # Gunakan pipeline dengan scaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        # Sesuaikan prefix parameter grid (contoh: 'clf__n_estimators')
        param_grid_prefixed = {f"clf__{k}": v for k, v in param_grid.items()}
        # Jalankan Grid Search
        grid = GridSearchCV(pipeline, param_grid_prefixed, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        # Hitung metrik
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log best hyperparams
        mlflow.log_params(grid.best_params_)

        # Log evaluation metrics
        mlflow.log_metric(f"{model_name}_accuracy", acc)
        mlflow.log_metric(f"{model_name}_precision", prec)
        mlflow.log_metric(f"{model_name}_recall", rec)
        mlflow.log_metric(f"{model_name}_f1_score", f1)

        # Simpan confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = f"{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Simpan classification report sebagai JSON
        report = classification_report(y_test, y_pred, output_dict=True)
        report_json_path = f"{model_name}_classification_report.json"
        with open(report_json_path, "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(report_json_path)

        # Simpan estimator sebagai HTML
        html_path = f"{model_name}_estimator.html"
        with open(html_path, "w") as f:
            f.write("<html><body>")
            f.write(f"<h1>{model_name} - Best Estimator</h1>")
            f.write(f"<pre>{str(best_model)}</pre>")
            f.write("</body></html>")
        mlflow.log_artifact(html_path)

        # Simpan model dengan log_model (menyimpan MLmodel, schema, preview, dll)
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(best_model, "best_model", signature=signature, input_example=X_test[:5])

print("✅ Model, metrik, schema, dan artifacts sudah dicatat ke MLflow lokal.")
