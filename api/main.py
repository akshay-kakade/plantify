import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

API_DIR = Path(__file__).resolve().parent
MODEL_DIR = API_DIR / "models"
TOMATO_MODEL_PATH = MODEL_DIR / "tomato_disease_mobilenetv3.pth"
POTATO_MODEL_PATH = MODEL_DIR / "1.keras"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
POTATO_CLASSES = ["Early Blight", "Late Blight", "Healthy"]

_tomato_model: nn.Module | None = None
_potato_model: Any | None = None

app = FastAPI(title="Verdant Plant Health API", version="1.0.0")
cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    # The API is public and does not use cookies or browser credentials.
    # Allow HTTPS-hosted frontends, including Vercel custom domains.
    allow_origin_regex=r"https://.*",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

TOMATO_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_tomato_model() -> nn.Module:
    global _tomato_model
    if _tomato_model is not None:
        return _tomato_model
    if not TOMATO_MODEL_PATH.exists():
        raise RuntimeError(f"Tomato model not found: {TOMATO_MODEL_PATH}")

    model = models.mobilenet_v3_small(weights=None)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1024, len(TOMATO_CLASSES)),
    )
    checkpoint = torch.load(TOMATO_MODEL_PATH, map_location=TORCH_DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    _tomato_model = model.to(TORCH_DEVICE).eval()
    return _tomato_model


def load_potato_model() -> Any:
    global _potato_model
    if _potato_model is not None:
        return _potato_model
    if not POTATO_MODEL_PATH.exists():
        raise RuntimeError(f"Potato model not found: {POTATO_MODEL_PATH}")

    import tensorflow as tf

    _potato_model = tf.keras.models.load_model(POTATO_MODEL_PATH, compile=False)
    return _potato_model


def read_image(contents: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Please upload a valid image file.") from error


def validate_upload(file: UploadFile, contents: bytes) -> None:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be smaller than 10 MB.")


def predict_tomato(image: Image.Image) -> dict[str, str]:
    model = load_tomato_model()
    tensor_image = TOMATO_TRANSFORM(image).unsqueeze(0).to(TORCH_DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor_image), dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)
    return {
        "predicted_class": TOMATO_CLASSES[predicted_index.item()],
        "confidence": f"{confidence.item() * 100:.2f}%",
    }


def predict_potato(image: Image.Image) -> dict[str, str]:
    model = load_potato_model()
    image_array = np.expand_dims(np.asarray(image, dtype=np.float32), 0)
    predictions = model.predict(image_array, verbose=0)[0]
    if predictions.shape[0] != len(POTATO_CLASSES):
        raise RuntimeError(
            f"Potato model returned {predictions.shape[0]} classes; expected {len(POTATO_CLASSES)}."
        )
    predicted_index = int(np.argmax(predictions))
    return {
        "predicted_class": POTATO_CLASSES[predicted_index],
        "confidence": f"{float(np.max(predictions)) * 100:.2f}%",
    }


@app.get("/")
def root():
    return {"message": "Verdant plant health API", "docs": "/docs", "models": ["tomato", "potato"]}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {"tomato": TOMATO_MODEL_PATH.exists(), "potato": POTATO_MODEL_PATH.exists()},
        "torch_device": str(TORCH_DEVICE),
    }


@app.get("/model-info")
def model_info():
    return {
        "tomato": {"architecture": "MobileNetV3 Small", "classes": TOMATO_CLASSES, "input_size": "256x256"},
        "potato": {"architecture": "Keras CNN", "classes": POTATO_CLASSES},
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), crop: str = Query("tomato", pattern="^(tomato|potato)$")):
    contents = await file.read()
    validate_upload(file, contents)
    image = read_image(contents)
    return predict_tomato(image) if crop == "tomato" else predict_potato(image)


@app.post("/predict/tomato")
async def predict_tomato_route(file: UploadFile = File(...)):
    contents = await file.read()
    validate_upload(file, contents)
    return predict_tomato(read_image(contents))


@app.post("/predict/potato")
async def predict_potato_route(file: UploadFile = File(...)):
    contents = await file.read()
    validate_upload(file, contents)
    return predict_potato(read_image(contents))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
