"""
Dataset Augmentation Script
15 images se 45+ images banao using augmentation
"""

import cv2
import numpy as np
from pathlib import Path
import random

def augment_image(img, num_augmentations=3):
    """Apply random augmentations to image."""
    augmented = []
    h, w = img.shape[:2]
    
    for _ in range(num_augmentations):
        aug = img.copy()
        
        # Random horizontal flip
        if random.random() > 0.5:
            aug = cv2.flip(aug, 1)
        
        # Random vertical flip
        if random.random() > 0.5:
            aug = cv2.flip(aug, 0)
        
        # Random rotation (-15 to +15 degrees)
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        aug = cv2.warpAffine(aug, M, (w, h))
        
        # Random brightness
        brightness = random.uniform(0.7, 1.3)
        aug = cv2.convertScaleAbs(aug, alpha=brightness, beta=0)
        
        # Random contrast
        contrast = random.uniform(0.8, 1.2)
        aug = cv2.convertScaleAbs(aug, alpha=contrast, beta=0)
        
        # Random noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 10, aug.shape).astype(np.uint8)
            aug = cv2.add(aug, noise)
        
        # Random crop and resize
        if random.random() > 0.6:
            crop_ratio = random.uniform(0.7, 0.9)
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
            y = random.randint(0, h - crop_h)
            x = random.randint(0, w - crop_w)
            aug = aug[y:y+crop_h, x:x+crop_w]
            aug = cv2.resize(aug, (w, h))
        
        augmented.append(aug)
    
    return augmented


def augment_dataset(input_dir, output_dir, augmentations_per_image=3):
    """Augment entire dataset."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get all images
    images_dir = input_path / 'images'
    labels_dir = input_path / 'labels'
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    print(f"Found {len(image_files)} images")
    
    count = 0
    for img_path in image_files:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Copy original
        cv2.imwrite(str(output_path / 'images' / img_path.name), img)
        
        # Copy corresponding label if exists
        label_path = labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            (output_path / 'labels' / label_path.name).write_text(
                label_path.read_text()
            )
        
        # Generate augmented versions
        augmented_images = augment_image(img, augmentations_per_image)
        
        for i, aug_img in enumerate(augmented_images):
            aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            cv2.imwrite(str(output_path / 'images' / aug_name), aug_img)
            
            # Copy same label for augmented image
            if label_path.exists():
                (output_path / 'labels' / f"{img_path.stem}_aug{i}.txt").write_text(
                    label_path.read_text()
                )
            
            count += 1
    
    print(f"Created {count} augmented images")
    print(f"Total images: {len(image_files) + count}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    # Augment train set
    print("=== Augmenting Train Set ===")
    augment_dataset(
        input_dir='dataset/train',
        output_dir='dataset/train_augmented',
        augmentations_per_image=3  # 15 images × 3 = 45 augmented
    )
    
    # Augment valid set
    print("\n=== Augmenting Valid Set ===")
    augment_dataset(
        input_dir='dataset/valid',
        output_dir='dataset/valid_augmented',
        augmentations_per_image=2
    )
    
    print("\n=== DONE ===")
    print("Now use augmented dataset for training")
