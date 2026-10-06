from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import pymupdf

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEXT_MODEL_PATH = BASE_DIR / "ml" / "text" / "distilbert_model_50k"
IMAGE_MODEL_PATH = (
    BASE_DIR / "ml" / "image" / "models" / "efficientnet_b0_new10k_best.pth"
)
FUSION_MODEL_PATH = (
    BASE_DIR / "ml" / "fusion" / "models" / "mlp_fusion_best.pth"
)


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="AI-Powered Multimodal Misinformation Detection API",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Load DistilBERT
# ============================================================

print("Loading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)

text_model = AutoModelForSequenceClassification.from_pretrained(
    TEXT_MODEL_PATH
)

text_model.to(device)
text_model.eval()

print("DistilBERT loaded.")


# ============================================================
# Load EfficientNet-B0
# ============================================================

print("Loading EfficientNet-B0...")

image_model = models.efficientnet_b0(weights=None)

image_model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 2)
)

checkpoint = torch.load(
    IMAGE_MODEL_PATH,
    map_location=device
)

if "model_state_dict" in checkpoint:
    image_model.load_state_dict(checkpoint["model_state_dict"])
else:
    image_model.load_state_dict(checkpoint)

image_model.to(device)
image_model.eval()

print("EfficientNet-B0 loaded.")


# ============================================================
# Image preprocessing
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Fusion MLP
# ============================================================

class FusionMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.network(x)


print("Loading fusion MLP...")

fusion_checkpoint = torch.load(
    FUSION_MODEL_PATH,
    map_location=device
)

fusion_model = FusionMLP()

if "model_state_dict" in fusion_checkpoint:
    fusion_model.load_state_dict(
        fusion_checkpoint["model_state_dict"]
    )
else:
    fusion_model.load_state_dict(fusion_checkpoint)

fusion_model.to(device)
fusion_model.eval()


# ============================================================
# Training statistics used during fusion training
# ============================================================

if "train_mean" in fusion_checkpoint:
    train_mean = fusion_checkpoint["train_mean"].to(device)
    train_std = fusion_checkpoint["train_std"].to(device)
else:
    raise RuntimeError(
        "train_mean and train_std were not found in the fusion checkpoint."
    )

print("Fusion MLP loaded.")


# ============================================================
# Helper: Convert prediction to readable result
# ============================================================

def format_prediction(probabilities):

    fake_probability = probabilities[0].item()
    true_probability = probabilities[1].item()

    predicted_class = 0 if fake_probability >= true_probability else 1

    label = "Fake / Misinformation" if predicted_class == 0 else "True / Real"

    confidence = max(
        fake_probability,
        true_probability
    )

    return {
        "prediction": label,
        "label": predicted_class,
        "confidence": round(confidence * 100, 2),
        "fake_probability": round(fake_probability * 100, 2),
        "true_probability": round(true_probability * 100, 2),
    }


# ============================================================
# Text feature extraction
# ============================================================

def extract_text_features(text):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = text_model.distilbert(
            **encoded
        )

        features = outputs.last_hidden_state[:, 0, :]

    return features


# ============================================================
# Text-only prediction
# ============================================================

def predict_text_only(text):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = text_model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    result = format_prediction(probabilities)

    result["mode"] = "text"

    return result


# ============================================================
# Image feature extraction
# ============================================================

def extract_image_features(image):

    image = image.convert("RGB")

    tensor = image_transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        features = image_model.features(tensor)

        features = image_model.avgpool(features)

        features = torch.flatten(
            features,
            start_dim=1
        )

    return features


# ============================================================
# Image-only prediction
# ============================================================

def predict_image_only(image):

    image = image.convert("RGB")

    tensor = image_transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        logits = image_model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    result = format_prediction(probabilities)

    result["mode"] = "image"

    return result


# ============================================================
# Multimodal prediction
# ============================================================

def predict_multimodal(text, image):

    text_features = extract_text_features(text)

    image_features = extract_image_features(image)

    combined_features = torch.cat(
        [text_features, image_features],
        dim=1
    )

    # Standardization using statistics from fusion training
    combined_features = (
        combined_features - train_mean
    ) / train_std

    with torch.no_grad():

        logits = fusion_model(
            combined_features
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    result = format_prediction(probabilities)

    result["mode"] = "multimodal"

    return result


# ============================================================
# PDF text extraction
# ============================================================

def extract_pdf_text(pdf_bytes):

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    extracted_text = []

    for page in document:

        text = page.get_text()

        if text:
            extracted_text.append(text)

    document.close()

    return "\n".join(extracted_text).strip()


# ============================================================
# PDF prediction
# ============================================================

def predict_pdf(pdf_bytes):

    extracted_text = extract_pdf_text(pdf_bytes)

    if not extracted_text:

        raise ValueError(
            "No extractable text was found in the PDF. "
            "Scanned/image-only PDFs are not supported yet."
        )

    result = predict_text_only(
        extracted_text
    )

    result["mode"] = "pdf"

    # Useful for debugging/testing
    result["extracted_text_length"] = len(
        extracted_text
    )

    return result


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI-Powered Multimodal Misinformation Detection API",
        "status": "running"
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(

    text: str = Form(""),

    image: UploadFile = File(None),

    pdf: UploadFile = File(None)
):

    text = text.strip()

    # --------------------------------------------------------
    # No input
    # --------------------------------------------------------

    if not text and image is None and pdf is None:

        return {
            "error": "Please provide text, an image, or a PDF."
        }


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if pdf is not None:

        if not pdf.filename.lower().endswith(".pdf"):

            return {
                "error": "The uploaded file is not a PDF."
            }

        pdf_bytes = await pdf.read()

        try:

            result = predict_pdf(
                pdf_bytes
            )

            return result

        except ValueError as e:

            return {
                "error": str(e)
            }


    # --------------------------------------------------------
    # Image + Text
    # --------------------------------------------------------

    if image is not None and text:

        image_bytes = await image.read()

        try:

            pil_image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

        except Exception:

            return {
                "error": "Invalid image file."
            }

        return predict_multimodal(
            text,
            pil_image
        )


    # --------------------------------------------------------
    # Image only
    # --------------------------------------------------------

    if image is not None:

        image_bytes = await image.read()

        try:

            pil_image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

        except Exception:

            return {
                "error": "Invalid image file."
            }

        return predict_image_only(
            pil_image
        )


    # --------------------------------------------------------
    # Text only
    # --------------------------------------------------------

    if text:

        return predict_text_only(
            text
        )