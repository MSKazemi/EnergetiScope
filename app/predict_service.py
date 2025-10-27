# predict_service.py
import os, uvicorn, numpy as np, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from typing import Dict, Any, List

from k8s_encode import K8sEncoder, _flat_row    # reuse your encoder utils
from models import InferenceRequest             # your schema

path ="/opt/local-path-provisioner/pvc-dde8a16d-5550-41b8-ac85-e75d5e49b7fc_energy_podpower-data"

ENC_PATH = os.getenv("ENCODER_PATH", path+"/encoder.joblib")
MOD_PATH = os.getenv("MODEL_PATH",   path+"/knn_energy.joblib")

app = FastAPI()
enc = K8sEncoder.load(ENC_PATH)
model = load(MOD_PATH)

class PredictOut(BaseModel):
    pred_energy_step_j: float
    workload_kind: str
    workload_name: str
    namespace: str
    spec_hash: str

@app.post("/predict", response_model=PredictOut)
def predict(ir: InferenceRequest):
    row = _flat_row(ir.model_dump())
    X, meta = enc.transform([row])
    y = float(model.predict(X)[0])
    m = meta.iloc[0].to_dict()
    return PredictOut(
        pred_energy_step_j=y,
        workload_kind=m["workload_kind"],
        workload_name=m["workload_name"],
        namespace=m["namespace"],
        spec_hash=m["_spec_hash"],
    )
if __name__ == "__main__":
    uvicorn.run("predict_service:app", host="0.0.0.0", port=8000)