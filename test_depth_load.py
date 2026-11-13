
from pathlib import Path
import torch
import sys

BASE_DIR = Path(__file__).resolve().parent

REPO_DIR = BASE_DIR / "depth_anything_v2_repo"

CHECKPOINT = (
    REPO_DIR / "checkpoints" / "depth_anything_v2_vits.pth"
)

sys.path.insert(0, str(REPO_DIR))

print("Loading Depth Anything V2 Small on CPU...", flush=True)

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

torch.set_num_threads(4)

from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}

model = DepthAnythingV2(**model_configs)

state_dict = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)

model = model.to("cpu")
model.eval()

print("SUCCESS: Depth Anything V2 Small loaded!", flush=True)
print("Device: CPU")
print("Model: Depth Anything V2 Small (ViT-S)")
