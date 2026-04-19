"""
app.py

FastAPI inference endpoint for plant disease classification.
Accepts a plant image and returns the predicted disease with confidence scores.

Swagger UI available at /docs after startup.

Local run:
    cd api
    uvicorn app:app --reload

HuggingFace Spaces: starts automatically via Dockerfile.
"""

import io
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

import timm
import torch.nn as nn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from huggingface_hub import hf_hub_download

# ─── CONFIG ───────────────────────────────────────────────────────────────────

HF_REPO_ID     = "Lusinen2004/plant-disease-classifier"
WEIGHTS_FILE   = "efficientnet_b0_best.pt"
CLASS_MAP_FILE = "class_mapping.json"
BACKBONE_NAME  = "efficientnet_b0"
IMAGE_SIZE     = 224
NUM_CLASSES    = 39
TOP_K          = 5      # how many top predictions to return

# ImageNet normalisation — must match what was used during training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── MODEL DEFINITION ─────────────────────────────────────────────────────────
# Reproduced here so the API has no dependency on src/
# (keeps the API folder self-contained and deployable independently)

class DiseaseClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        with torch.no_grad():
            dummy       = torch.zeros(1, 3, 224, 224)
            feature_dim = self.backbone(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ─── STARTUP: LOAD MODEL ──────────────────────────────────────────────────────

def load_model_from_hub() -> tuple[DiseaseClassifier, dict]:
    """
    Download weights and class mapping from HuggingFace Hub on startup.

    Why HF Hub and not bundling weights in the repo?
        Model weights are binary blobs that don't belong in git history.
        HF Hub is purpose-built for model artefacts, gives you a free
        CDN, and lets you update weights without changing code.

    Returns:
        model       : DiseaseClassifier ready for inference
        idx_to_class: dict mapping int index -> disease name string
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    # Download from Hub (cached after first download)
    print("  Downloading weights from HuggingFace Hub...")
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=WEIGHTS_FILE)
    map_path     = hf_hub_download(repo_id=HF_REPO_ID, filename=CLASS_MAP_FILE)
    print(f"  Weights : {weights_path}")
    print(f"  Map     : {map_path}")

    # Load class mapping
    with open(map_path) as f:
        raw = json.load(f)
    idx_to_class = {int(k): v for k, v in raw.items()}
    print(f"  Classes : {len(idx_to_class)}")

    # Build model architecture
    backbone = timm.create_model(
        BACKBONE_NAME,
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )
    model = DiseaseClassifier(
        backbone=backbone,
        num_classes=NUM_CLASSES,
    )

    # Load saved weights
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    print("  Model loaded and ready.")

    return model, idx_to_class, device


# Load once at startup — not per request
print("=== Loading model ===")
MODEL, IDX_TO_CLASS, DEVICE = load_model_from_hub()

# ─── TRANSFORMS ───────────────────────────────────────────────────────────────

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─── FASTAPI APP ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Plant Disease Classifier",
    description = (
        "Classifies plant diseases from leaf images regardless of host plant species. "
        "Upload an image and receive the predicted disease with confidence scores.\n\n"
        "**Model**: EfficientNet-B0 fine-tuned on 8k plant disease images.\n\n"
        "**Output**: Disease name only (e.g. 'late blight', not 'tomato late blight')."
    ),
    version     = "1.0.0",
)

# ─── RESPONSE SCHEMA ──────────────────────────────────────────────────────────

class Prediction(BaseModel):
    disease      : str
    confidence   : float
    top_k        : list[dict]

    class Config:
        json_schema_extra = {
            "example": {
                "disease"    : "late blight",
                "confidence" : 0.9231,
                "top_k": [
                    {"disease": "late blight",  "confidence": 0.9231},
                    {"disease": "early blight", "confidence": 0.0412},
                    {"disease": "leaf spot",    "confidence": 0.0187},
                    {"disease": "anthracnose",  "confidence": 0.0091},
                    {"disease": "rust",         "confidence": 0.0043},
                ],
            }
        }


class HealthResponse(BaseModel):
    status      : str
    model       : str
    num_classes : int


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns model status and configuration. Use to verify the API is running.",
)
def health_check():
    return HealthResponse(
        status      = "ok",
        model       = BACKBONE_NAME,
        num_classes = NUM_CLASSES,
    )


@app.post(
    "/predict",
    response_model=Prediction,
    summary="Predict plant disease",
    description=(
        "Upload a JPG or PNG image of a plant or leaf. "
        "Returns the predicted disease name and top-5 confidence scores. "
        "The prediction is plant-agnostic — the same disease is returned "
        "regardless of which plant species is in the image."
    ),
)
async def predict(file: UploadFile = File(..., description="Plant leaf image (JPG or PNG)")):
    """
    Main inference endpoint.

    Steps:
      1. Validate the uploaded file is a readable image
      2. Apply the same transforms used during validation
      3. Run a forward pass through the model
      4. Return the top-1 prediction and top-K confidence scores
    """
    # ── Validate file type ────────────────────────────────────────────────
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(
            status_code = 415,
            detail      = f"Unsupported file type '{file.content_type}'. Upload JPG or PNG.",
        )

    # ── Read and decode image ─────────────────────────────────────────────
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code = 400,
            detail      = "Could not decode image. Ensure the file is a valid JPG or PNG.",
        )

    # ── Preprocess ────────────────────────────────────────────────────────
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    # unsqueeze(0) adds the batch dimension: [3, 224, 224] → [1, 3, 224, 224]

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = MODEL(tensor)                        # [1, num_classes]
        probs  = F.softmax(logits, dim=1).squeeze(0)  # [num_classes]

    # ── Top-K results ─────────────────────────────────────────────────────
    top_probs, top_indices = torch.topk(probs, k=TOP_K)
    top_k_results = [
        {
            "disease"    : IDX_TO_CLASS[idx.item()],
            "confidence" : round(prob.item(), 4),
        }
        for prob, idx in zip(top_probs, top_indices)
    ]

    return Prediction(
        disease    = top_k_results[0]["disease"],
        confidence = top_k_results[0]["confidence"],
        top_k      = top_k_results,
    )