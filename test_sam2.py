
from pathlib import Path
from sam2.build_sam import build_sam2

BASE_DIR = Path(__file__).resolve().parent

checkpoint_path = BASE_DIR / "sam2_repo/checkpoints" / "sam2.1_hiera_tiny.pt"

print("Loading SAM 2.1 Tiny on CPU...", flush=True)

model = build_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    str(checkpoint_path),
    device="cpu",
)

print("SUCCESS: SAM 2.1 Tiny model loaded!", flush=True)
