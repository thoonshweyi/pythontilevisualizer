
from pathlib import Path
from contextlib import asynccontextmanager
import base64
import threading

import cv2
import numpy as np
import torch

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ==================================================
# 1. PROJECT PATHS
# ==================================================

REPO_DIR = Path(__file__).resolve().parent

SAM2_REPO_DIR = REPO_DIR / "sam2_repo"

CHECKPOINT_PATH = (
    SAM2_REPO_DIR
    / "checkpoints"
    / "sam2.1_hiera_tiny.pt"
)

# IMPORTANT:
# Use the same Hydra config name as your working test script.
CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_t.yaml"

DEVICE = "cpu"

predictor = None
sam_lock = threading.Lock()


# ==================================================
# 2. LOAD SAM MODEL ONCE
# ==================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    print("1. Loading SAM 2.1 Tiny...", flush=True)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Run from the SAM repository directory so Hydra can
    # resolve configs/sam2.1/sam2.1_hiera_t.yaml correctly.
    import os

    original_cwd = os.getcwd()

    try:
        os.chdir(SAM2_REPO_DIR)

        model = build_sam2(
            CONFIG_PATH,
            str(CHECKPOINT_PATH),
            device=DEVICE,
        )
    finally:
        os.chdir(original_cwd)

    predictor = SAM2ImagePredictor(model)

    print("SUCCESS: SAM 2.1 Tiny loaded!", flush=True)

    try:
        yield
    finally:
        predictor = None
        print("SAM model unloaded.", flush=True)


app = FastAPI(
    title="2D Tile Visualizer API",
    lifespan=lifespan,
)


# ==================================================
# 3. CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# 4. HEALTH CHECK
# ==================================================

@app.get("/api/health")
def health_check():
    return {
        "success": True,
        "model_loaded": predictor is not None,
        "device": DEVICE,
    }


# ==================================================
# 5. MASK API
# ==================================================

@app.post("/api/mask")
async def create_mask(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
):
    print(f"Received click: ({x}, {y})", flush=True)

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="SAM model is not loaded",
        )

    # ----------------------------------------------
    # Read uploaded image
    # ----------------------------------------------

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty",
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Cannot decode uploaded image",
        )

    height, width = image_bgr.shape[:2]

    print(
        f"Image size: {width} x {height}",
        flush=True,
    )

    # ----------------------------------------------
    # Validate click coordinates
    # ----------------------------------------------

    if not (0 <= x < width and 0 <= y < height):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Click point is outside the image",
                "image_width": width,
                "image_height": height,
            },
        )

    # ----------------------------------------------
    # Convert BGR to RGB
    # Same as your working test script
    # ----------------------------------------------

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # ----------------------------------------------
    # Run segmentation
    # ----------------------------------------------

    try:
        with sam_lock:
            print("Setting image...", flush=True)

            predictor.set_image(image_rgb)

            print("Running segmentation...", flush=True)

            with torch.inference_mode():
                masks, scores, logits = predictor.predict(
                    point_coords=np.array(
                        [[x, y]],
                        dtype=np.float32,
                    ),
                    point_labels=np.array(
                        [1],
                        dtype=np.int32,
                    ),
                    multimask_output=True,
                )

            best_index = int(np.argmax(scores))
            best_mask = masks[best_index]

            best_score = float(scores[best_index])

    except Exception as exc:
        print(
            f"Segmentation error: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {exc}",
        ) from exc

    # ----------------------------------------------
    # Convert mask to binary PNG
    # White = selected region
    # Black = background
    # ----------------------------------------------

    mask_image = (
        best_mask.astype(np.uint8) * 255
    )

    success, encoded_mask = cv2.imencode(
        ".png",
        mask_image,
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode mask PNG",
        )

    mask_base64 = base64.b64encode(
        encoded_mask.tobytes()
    ).decode("utf-8")

    print(
        f"Segmentation complete. Score: {best_score:.4f}",
        flush=True,
    )

    # ----------------------------------------------
    # Response for React + OpenCV.js
    # ----------------------------------------------

    return {
        "success": True,
        "image_width": width,
        "image_height": height,
        "mask": mask_base64,
        "mask_format": "png_base64",
        "mask_encoding": "binary",
        "score": round(best_score, 4),
        "prompt": {
            "x": x,
            "y": y,
        },
    }

# uvicorn main:app --reload