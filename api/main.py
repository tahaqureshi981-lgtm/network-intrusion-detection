import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="Network Intrusion Detection API",
    description="Detects malicious network activity using ML",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"

# ── Load artifacts ────────────────────────────────────────────────────────────
try:
    model         = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    scaler        = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    encoders      = joblib.load(f"{MODELS_DIR}/encoders.pkl")
    label_encoder = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")
    feature_names = joblib.load(f"{MODELS_DIR}/feature_names.pkl")
    classes       = joblib.load(f"{MODELS_DIR}/classes.pkl")
    model_name    = joblib.load(f"{MODELS_DIR}/best_model_name.pkl")
    print(f"✅ Loaded model: {model_name}")
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# ── Risk levels ───────────────────────────────────────────────────────────────
RISK_LEVEL = {
    "Normal": "LOW",
    "Probe":  "MEDIUM",
    "DoS":    "HIGH",
    "R2L":    "HIGH",
    "U2R":    "CRITICAL"
}

ATTACK_DESC = {
    "Normal": "Legitimate network traffic — no threat detected.",
    "DoS":    "Denial of Service attack — attempt to overwhelm system resources.",
    "Probe":  "Reconnaissance activity — attacker scanning for vulnerabilities.",
    "R2L":    "Remote to Local attack — unauthorized access attempt from remote machine.",
    "U2R":    "User to Root attack — privilege escalation attempt detected."
}

# ── Request schema ────────────────────────────────────────────────────────────
class NetworkConnection(BaseModel):
    duration: float             = Field(0, description="Connection duration in seconds")
    protocol_type: str          = Field("tcp", description="tcp / udp / icmp")
    service: str                = Field("http", description="Network service (http, ftp, ssh...)")
    flag: str                   = Field("SF", description="Connection status flag")
    src_bytes: float            = Field(0, description="Bytes sent from source")
    dst_bytes: float            = Field(0, description="Bytes sent to destination")
    land: int                   = Field(0, description="1 if src/dst host/port are same")
    wrong_fragment: float       = Field(0)
    urgent: float               = Field(0)
    hot: float                  = Field(0)
    num_failed_logins: float    = Field(0)
    logged_in: int              = Field(0)
    num_compromised: float      = Field(0)
    root_shell: float           = Field(0)
    su_attempted: float         = Field(0)
    num_root: float             = Field(0)
    num_file_creations: float   = Field(0)
    num_shells: float           = Field(0)
    num_access_files: float     = Field(0)
    num_outbound_cmds: float    = Field(0)
    is_host_login: int          = Field(0)
    is_guest_login: int         = Field(0)
    count: float                = Field(1)
    srv_count: float            = Field(1)
    serror_rate: float          = Field(0)
    srv_serror_rate: float      = Field(0)
    rerror_rate: float          = Field(0)
    srv_rerror_rate: float      = Field(0)
    same_srv_rate: float        = Field(1)
    diff_srv_rate: float        = Field(0)
    srv_diff_host_rate: float   = Field(0)
    dst_host_count: float       = Field(255)
    dst_host_srv_count: float   = Field(255)
    dst_host_same_srv_rate: float     = Field(1)
    dst_host_diff_srv_rate: float     = Field(0)
    dst_host_same_src_port_rate: float = Field(0)
    dst_host_srv_diff_host_rate: float = Field(0)
    dst_host_serror_rate: float       = Field(0)
    dst_host_srv_serror_rate: float   = Field(0)
    dst_host_rerror_rate: float       = Field(0)
    dst_host_srv_rerror_rate: float   = Field(0)

    class Config:
        json_schema_extra = {
            "example": {
                "duration": 0, "protocol_type": "tcp", "service": "http",
                "flag": "SF", "src_bytes": 232, "dst_bytes": 8153,
                "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
                "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
                "root_shell": 0, "su_attempted": 0, "num_root": 0,
                "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
                "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
                "count": 8, "srv_count": 8, "serror_rate": 0, "srv_serror_rate": 0,
                "rerror_rate": 0, "srv_rerror_rate": 0, "same_srv_rate": 1,
                "diff_srv_rate": 0, "srv_diff_host_rate": 0, "dst_host_count": 255,
                "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1,
                "dst_host_diff_srv_rate": 0, "dst_host_same_src_port_rate": 0,
                "dst_host_srv_diff_host_rate": 0, "dst_host_serror_rate": 0,
                "dst_host_srv_serror_rate": 0, "dst_host_rerror_rate": 0,
                "dst_host_srv_rerror_rate": 0
            }
        }


# ── Helper ────────────────────────────────────────────────────────────────────
def build_features(conn: NetworkConnection) -> np.ndarray:
    data = conn.dict()

    # Encode categorical features
    for col in ["protocol_type", "service", "flag"]:
        le = encoders[col]
        val = str(data[col])
        if val not in le.classes_:
            val = le.classes_[0]
        data[col] = le.transform([val])[0]

    df = pd.DataFrame([data])[feature_names]
    return scaler.transform(df)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Network Intrusion Detection API", "model": model_name}


@app.get("/health")
def health():
    return {"status": "ok", "model": model_name, "classes": classes}


@app.post("/predict")
def predict(conn: NetworkConnection):
    try:
        features   = build_features(conn)
        pred_enc   = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0]
        pred_label = label_encoder.inverse_transform([pred_enc])[0]

        confidence = float(pred_proba[pred_enc])
        prob_dict  = {cls: round(float(p), 4)
                      for cls, p in zip(label_encoder.classes_, pred_proba)}

        return {
            "prediction":   pred_label,
            "confidence":   round(confidence, 4),
            "risk_level":   RISK_LEVEL.get(pred_label, "UNKNOWN"),
            "description":  ATTACK_DESC.get(pred_label, ""),
            "probabilities": prob_dict,
            "model_used":   model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info():
    return {
        "model_name":    model_name,
        "classes":       classes,
        "n_features":    len(feature_names),
        "feature_names": feature_names
    }