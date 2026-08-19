import os
import zipfile
import mlflow
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import deepxde as dde

app = FastAPI(title="Intercooler PINN Digital Twin API")

MODEL_DIR = "downloaded_model"
MODEL_ZIP = "pinn_model_files.zip"

class SensorData(BaseModel):
    time_hours: float
    current_u_value: float

@app.on_event("startup")
def load_model_from_mlflow():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    
    mlflow.set_tracking_uri('https://dagshub.com/saad.tatakae/Intercooler-LSTM.mlflow')
    
    
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id="m_e0f16bb3ce954b55999012455b0285f1",
        artifact_path="final_model/pinn_model_files.zip",
        dst_path=MODEL_DIR
    )
    
    
    zip_path = os.path.join(MODEL_DIR, "final_model", MODEL_ZIP)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    print("Model successfully loaded from MLflow!")

@app.post("/predict")
def predict_degradation(data: SensorData):
    try:
        
        t_scaled = np.array([[data.time_hours / 1000.0]])
        
        
        net = dde.nn.FNN([1] + [64] * 3 + [1], "tanh", "Glorot normal")
        model = dde.Model(None, net)
        model.compile("adam", lr=0.001)
        
        
        ckpt_path = tf.train.latest_checkpoint(MODEL_DIR)
        if ckpt_path:
            model.restore(ckpt_path, verbose=0)
            
    
        prediction = model.predict(t_scaled)
        predicted_u = float(prediction[0][0]) * 100.0
        
        return {
            "status": "success",
            "input_time_hours": data.time_hours,
            "predicted_u_value_wm2k": predicted_u
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))