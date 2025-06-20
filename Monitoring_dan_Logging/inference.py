import requests
import json

# URL untuk serving model
URL = "http://localhost:5001/invocations"

# Path ke input_example.json (relatif dari file ini)
input_example_path = "Membangun_model/mlartifacts/367316111141457261/2b30f73e29a949248a3a359cfbffa1e7/artifacts/XGBoost_mlflow_model/input_example.json"

# Load data JSON
with open(input_example_path, "r") as f:
    data = json.load(f)

# Kirim request
response = requests.post(URL, json=data)

# Tampilkan hasil
print("Prediction response:", response.text)
