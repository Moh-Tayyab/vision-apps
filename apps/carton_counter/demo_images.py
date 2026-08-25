"""Generate synthetic carton images for demo purposes."""

import os
import random
import cv2
import numpy as np


def generate_carton_image(
    num_cartons: int = 12,
    width: int = 800,
    height: int = 600,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic image with carton-like boxes on a pallet."""
    random.seed(seed)
    np.random.seed(seed)

    img = np.ones((height, width, 3), dtype=np.uint8) * 240

    pallet_y = int(height * 0.75)
    cv2.rectangle(img, (100, pallet_y), (width - 100, height - 50), (120, 100, 80), -1)
    cv2.rectangle(img, (80, height - 50), (width - 80, height - 30), (100, 80, 60), -1)

    rows = 3
    cols = 4
    box_w = 140
    box_h = 90
    start_x = (width - cols * box_w) // 2 + random.randint(-20, 20)
    start_y = pallet_y - rows * box_h + 10

    colors = [
        (180, 160, 120), (160, 140, 100), (200, 180, 140),
        (170, 150, 110), (190, 170, 130), (150, 130, 90),
    ]

    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= num_cartons:
                break
            x = start_x + col * box_w + random.randint(-8, 8)
            y = start_y + row * box_h + random.randint(-5, 5)
            w = box_w - 15 + random.randint(-10, 10)
            h = box_h - 10 + random.randint(-8, 8)

            color = colors[count % len(colors)]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (80, 60, 40), 2)

            label = f"C{count + 1}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(
                img, label,
                (x + w // 2 - tw // 2, y + h // 2 + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 30, 20), 1,
            )
            count += 1

    cv2.putText(
        img, f"Pallet View - {num_cartons} Cartons",
        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2,
    )

    return img


def generate_multi_view_images(num_cartons: int = 12):
    """Generate front, side, and top views for multi-angle demo."""
    front = generate_carton_image(num_cartons, seed=42)
    side = generate_carton_image(num_cartons, seed=123)
    top = generate_carton_image(num_cartons, seed=456)
    return front, side, top


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "demo_images")
    os.makedirs(out_dir, exist_ok=True)

    front, side, top = generate_multi_view_images(12)
    cv2.imwrite(os.path.join(out_dir, "front.jpg"), front)
    cv2.imwrite(os.path.join(out_dir, "side.jpg"), side)
    cv2.imwrite(os.path.join(out_dir, "top.jpg"), top)

    single = generate_carton_image(8, seed=99)
    cv2.imwrite(os.path.join(out_dir, "single.jpg"), single)

    print(f"Demo images saved to {out_dir}/")
    print("  front.jpg, side.jpg, top.jpg (multi-angle)")
    print("  single.jpg (single detection)")
