from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import mlflow
import os

# 🛡️ Hindari konflik jika ada file bernama 'mlflow.py'
need_restart = False
if os.path.exists("mlflow.py"):
    os.rename("mlflow.py", "mlflow_shadow.py")
    need_restart = True
if os.path.exists("mlflow.pyc"):
    os.remove("mlflow.pyc")
    need_restart = True

if need_restart:
    print("A file named 'mlflow.py' or 'mlflow.pyc' was found and renamed/removed.")
    print("Please RESTART the kernel and rerun this cell to avoid import errors.")
    sys.exit()

# Setup tracking
mlflow.set_tracking_uri("file:///content/drive/MyDrive/MSML/Proyek_Akhir/mlruns")
mlflow.set_experiment("Dropout_Prediction_MultiModel_Tuning")

# Create models and param grids
models_with_params = {
    'XGBoost': (
        XGBClassifier(eval_metric='logloss'),
        {
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [100, 200]
        }
    ),
    'Random Forest': (
        RandomForestClassifier(),
        {
            'n_estimators': [100, 200],
            'max_depth': [None, 10]
        }
    ),
    'Logistic Regression': (
        LogisticRegression(max_iter=1000),
        {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear']
        }
    ),
    'Decision Tree': (
        DecisionTreeClassifier(),
        {
            'max_depth': [5, 10, None],
            'criterion': ['gini', 'entropy']
        }
    ),
    'SVC': (
        SVC(),
        {
            'C': [0.1, 1.0],
            'kernel': ['linear', 'rbf']
        }
    )
}

# Jalankan satu run untuk semua model
with mlflow.start_run(run_name="All_Models_Tuning"):
    for model_name, (model, param_grid) in models_with_params.items():

        grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Logging params (dengan prefix model_name)
        for param, value in grid.best_params_.items():
            mlflow.log_param(f"{model_name}_{param}", value)

        # Logging metrics
        mlflow.log_metric(f"{model_name}_accuracy", acc)
        mlflow.log_metric(f"{model_name}_precision", prec)
        mlflow.log_metric(f"{model_name}_recall", rec)
        mlflow.log_metric(f"{model_name}_f1_score", f1)

        # Simpan dan log model artifact
        path = f"mlruns/models/{model_name.replace(' ', '_')}_tuned.pkl"
        joblib.dump(best_model, path)
        mlflow.log_artifact(path)

        print(f"✅ {model_name} selesai ditraining dan dilog.")

print("🎉 Semua model telah dituning dan dilog dalam satu run.")