from fastapi import FastAPI, HTTPException
from .settings import settings
from fastapi.middleware.cors import CORSMiddleware
from app.predict_digit import predict




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dariomesasmarti.com", 
        "https://www.dariomesasmarti.com",
        "http://localhost:4200"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Especifica solo los métodos que necesitas
    allow_headers=["Content-Type", "X-API-Key"],  # Especifica solo los headers que usas
)


# Test para ver si sigue funcionando
@app.get("/status")
async def status():
    return {"status" : "ok"}


@app.post("/predict")
# Endpoint para consultar qué precide el modelo
async def predict_endpoint(b64: str):
    try:
        return predict(b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint para añadir una imagen a la predicción del modelo
@app.post("/add-sample")
# Se ha de pasar la imagen en base 64, el resultado correcto y la api key
async def add_sample():
    # Se ha de comprobar que la api key es correcta
    # Se consulta con el modelo para ver qué número cree que es
    # Devolver la lista de los números que cree que pueden ser
    return {"status":"ok", "prediction":"5"}



@app.get("/")
async def root(id):
    if id != settings.api_key:
        raise HTTPException(401)
    return {"message" : "Hello world"}