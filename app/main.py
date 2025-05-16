from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
from app.predict_digit import predict as model_predict
from pydantic import BaseModel
from app.add_sample import save_sample, retrain_model
from app.settings import settings  # Importar las configuraciones


class SampleInput(BaseModel):
    b64: str
    label: int


# Ruta absoluta a model.h5
APP_DIR     = Path(__file__).resolve().parent
MODEL_PATH  = Path("model.h5")


# Dependencia para validar la API key
async def validate_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key inválida"
        )
    return x_api_key


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dariomesasmarti.com",
        "https://www.dariomesasmarti.com",
        "http://localhost:4200",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
    allow_credentials=True,
)

api = APIRouter(prefix="/api")

# 1) Ping /api/ping - Ahora con autenticación
@api.get("/ping", tags=["util"], dependencies=[Depends(validate_api_key)])
async def ping():
    return {"status": "ok"}

# 2) Model status /api/model - Ahora con autenticación
@api.get("/model", tags=["util"], dependencies=[Depends(validate_api_key)])
async def model_status():
    exists = MODEL_PATH.exists()
    return {
        "model_exists": exists,
        "path": str(MODEL_PATH),
        "size_bytes": MODEL_PATH.stat().st_size if exists else 0,
    }

# 3) Predict /api/predict - Ahora con autenticación
@api.post("/predict", tags=["model"], dependencies=[Depends(validate_api_key)])
async def predict_endpoint(b64: str):
    try:
        return model_predict(b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 4) Add sample /api/add-sample - Sin cambios porque tiene TODO
@api.post("/add-sample", tags=["model"])
async def add_sample_endpoint(sample: SampleInput):
    #TODO
    return {"status": "ok", "log": ""}
    try:
        saved_path = save_sample(sample.b64, sample.label)
        retrain_output = retrain_model()
        return {
            "status": "ok",
            "saved": str(saved_path),
            "retrain_log": retrain_output
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 5) Train model /api/train - Sin cambios porque tiene TODO
@api.post("/train", tags=["model"])
async def train_model_endpoint():
    #TODO
    return {"status": "ok", "log": ""}
    try:
        from app.add_sample import retrain_model
        log = retrain_model()
        return {"status": "ok", "log": log}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api)