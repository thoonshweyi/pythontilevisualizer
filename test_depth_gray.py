
from pathlib import Path
import sys
import cv2
import torch
import numpy as np
import time

BASE_DIR = Path(__file__).resolve().parent

# Depth Anything V2 Repository
REPO_DIR = BASE_DIR / "depth_anything_v2_repo"

# Checkpoint
checkpoint_path = (
    REPO_DIR / "checkpoints" / "depth_anything_v2_vits.pth"
)

# Import from repository
sys.path.insert(0, str(REPO_DIR))

DEVICE = "cpu"

torch.set_num_threads(4)

print("Loading Depth Anything V2 Small on CPU...", flush=True)

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}

model = DepthAnythingV2(**model_configs)

state_dict = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

print("SUCCESS: Depth Anything V2 Small loaded!", flush=True)

# Test image
image_path = BASE_DIR / "static/images/simplebedroom.jpg"

raw_img = cv2.imread(str(image_path))

if raw_img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

print(f"Image size: {raw_img.shape}", flush=True)

# Depth inference
print("Running depth estimation...", flush=True)

start_time = time.time()

with torch.inference_mode():
    depth = model.infer_image(raw_img)

elapsed = time.time() - start_time

print(f"Depth shape: {depth.shape}")
print(f"Depth min: {depth.min():.4f}")
print(f"Depth max: {depth.max():.4f}")
print(f"Inference time: {elapsed:.2f} seconds")

# Normalize depth
depth_normalized = cv2.normalize(
    depth,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# Save grayscale depth map
cv2.imwrite(
    str(BASE_DIR / "depth_gray.png"),
    depth_normalized
)

# Save color depth map
depth_color = cv2.applyColorMap(
    depth_normalized,
    cv2.COLORMAP_INFERNO
)

cv2.imwrite(
    str(BASE_DIR / "depth_color.png"),
    depth_color
)

# Save raw depth values
np.save(
    str(BASE_DIR / "depth_raw.npy"),
    depth
)

print("SUCCESS: Depth maps generated!", flush=True)
print(f"Output folder: {BASE_DIR}", flush=True)
