
from pathlib import Path

import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Project paths
REPO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = REPO_DIR

IMAGE_PATH = PROJECT_DIR / "static/images" / "simplebedroom.jpg"
CHECKPOINT_PATH = REPO_DIR / "sam2_repo/checkpoints" / "sam2.1_hiera_tiny.pt"
CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Change these coordinates to a point INSIDE the wall
POINT_X = 356
POINT_Y = 74


def main():
    print("1. Reading room image...", flush=True)

    image_bgr = cv2.imread(str(IMAGE_PATH))

    if image_bgr is None:
        raise FileNotFoundError(
            f"Cannot read image: {IMAGE_PATH}"
        )

    height, width = image_bgr.shape[:2]

    print(f"Image size: {width} x {height}")

    if not (0 <= POINT_X < width and 0 <= POINT_Y < height):
        raise ValueError(
            f"Point ({POINT_X}, {POINT_Y}) is outside the image"
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("2. Loading SAM 2.1 Tiny...", flush=True)

    model = build_sam2(
        CONFIG_PATH,
        str(CHECKPOINT_PATH),
        device="cpu",
    )

    predictor = SAM2ImagePredictor(model)

    print("3. Setting image...", flush=True)
    predictor.set_image(image_rgb)

    print(
        f"4. Segmenting point ({POINT_X}, {POINT_Y})...",
        flush=True,
    )

    masks, scores, logits = predictor.predict(
        point_coords=np.array(
            [[POINT_X, POINT_Y]],
            dtype=np.float32,
        ),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )

    # Choose the mask with the highest predicted score
    best_index = int(np.argmax(scores))
    best_mask = masks[best_index]

    # Save binary mask: white = selected region, black = background
    mask_image = (best_mask.astype(np.uint8) * 255)

    mask_path = PROJECT_DIR / "test_images" / "wall_mask.png"
    cv2.imwrite(str(mask_path), mask_image)

    # Create an overlay to inspect the selected region
    overlay = image_bgr.copy()
    selected = best_mask.astype(bool)

    green = np.zeros_like(image_bgr)
    green[:, :, 1] = 255

    overlay[selected] = (
        0.5 * image_bgr[selected] +
        0.5 * green[selected]
    ).astype(np.uint8)

    # Red dot marks the prompt point
    cv2.circle(
        overlay,
        (POINT_X, POINT_Y),
        7,
        (0, 0, 255),
        -1,
    )

    overlay_path = PROJECT_DIR / "test_images" / "wall_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    print(f"Best mask score: {scores[best_index]:.4f}")
    print(f"Mask saved: {mask_path}")
    print(f"Overlay saved: {overlay_path}")
    print("SUCCESS: Segmentation completed!", flush=True)


if __name__ == "__main__":
    main()
