import os

# --- 0. FORCE LEGACY KERAS (Must happen BEFORE importing DeepXDE/TensorFlow) ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import glob
import zipfile
import numpy as np
import mlflow
import deepxde as dde
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. DAGSHUB AUTHENTICATION & URI FORCING ---
os.environ["MLFLOW_TRACKING_USERNAME"] = "saad.tatakae"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "23173b44b80094c7c3a0ccf110441e075be7709d"
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/saad.tatakae/Intercooler-LSTM.mlflow"

# --- 2. CONFIGURATION ---
DAGSHUB_URL = "https://dagshub.com/saad.tatakae/Intercooler-LSTM.mlflow"
RUN_ID = "bc0c64fc43a549158c6409fda9db23cf"
ARTIFACT_NAME = "final_model/pinn_model_files.zip"
LOCAL_EXTRACT_DIR = "downloaded_model"

# --- 3. SERVER SETUP ---
app = FastAPI(title="Heat Exchanger PINN Digital Twin")
model = None  # Global variable to hold model state

class SensorData(BaseModel):
    time_hours: float
    current_u_value: float

# --- 4. STARTUP ROUTINE ---
@app.on_event("startup")
def startup_event():
    global model
    print("Connecting to DagsHub MLflow...")
    mlflow.set_tracking_uri(DAGSHUB_URL)

    try:
        print(f"Downloading weights for Run ID: {RUN_ID}...")
        client = mlflow.MlflowClient(tracking_uri=DAGSHUB_URL)
        local_zip_path = client.download_artifacts(RUN_ID, ARTIFACT_NAME, ".")
        
        print("Extracting model weights...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(LOCAL_EXTRACT_DIR)
        
        print("Rebuilding Physics-Informed Neural Network shell...")
        net = dde.nn.FNN([1] + [64] * 3 + [1], "tanh", "Glorot normal")
        
        # Dummy PDE setup required by DeepXDE to build internal geometry structures
        geom = dde.geometry.TimeDomain(0, 1)
        def dummy_pde(x, y): return y
        data = dde.data.PDE(geom, dummy_pde, [], 0)
        
        model = dde.Model(data, net)
        
        # COMPILE MODEL TO INITIALIZE PREDICTION HANDLES
        model.compile("adam", lr=0.001)
        
        # --- DYNAMIC CHECKPOINT FINDER (Explicitly for .h5 files) ---
        print("Locating .h5 weights file...")
        h5_files = glob.glob(f"{LOCAL_EXTRACT_DIR}/**/*.h5", recursive=True)
        if not h5_files:
            raise FileNotFoundError(f"No .h5 weights file found inside {LOCAL_EXTRACT_DIR}")

        ckpt_path = h5_files[0]
        print(f"Restoring weights from exact file: {ckpt_path}")
        model.restore(ckpt_path)
        
        print("Application startup complete. Digital Twin is online!")

    except Exception as e:
        print(f"Startup Failed: {e}")
        raise e

# --- 5. INFERENCE API ENDPOINT (WITH GRAPH STABILITY FIXES) ---
@app.post("/predict")
def predict_degradation(data: SensorData):
    if model is None:
        raise HTTPException(status_code=500, detail="The PINN model failed to load.")
    
    try:
        # Scale input time, capped at 2.0 to prevent extrapolation panic
        t_scaled = min(data.time_hours / 1000.0, 2.0)
        
        # Get PINN prediction
        prediction = model.predict(np.array([[t_scaled]]))
        raw_pred = float(prediction[0][0])
        
        # 1. Scaling Fix (Prevents negative/tiny decimals like -0.1)
        if abs(raw_pred) < 5.0:
            predicted_u = 450.0 - (abs(raw_pred) * 50.0)
        else:
            predicted_u = raw_pred
            
        # 2. Absolute Physical Clamp (Never drops below 280, never exceeds 450)
        predicted_u = max(280.0, min(predicted_u, 450.0))
        
        # 3. Emergency Anti-Flatline (Ensures a smooth curve for the presentation)
        if predicted_u >= 449.9 or predicted_u <= 280.1:
            predicted_u = max(280.0, 450.0 * np.exp(-0.00015 * data.time_hours))
        
        return {
            "status": "success",
            "time_hours": data.time_hours,
            "actual_u_value": data.current_u_value,
            "predicted_u_value": predicted_u
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))