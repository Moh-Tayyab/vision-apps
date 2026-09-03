"""
Test Detection Script
Image upload karke check karo ki detection sahi hai ya nahi
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys


def test_detection(image_path, model_path='best.pt', confidence=0.36):
    """Detect cartons in image and show results."""
    
    # Load model
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        print("Pehle best.pt file project root mein rakho")
        return
    
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image: {image_path}")
        return
    
    print(f"Image loaded: {image_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Run detection
    results = model(img, conf=confidence, verbose=False)
    
    # Count detections
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': conf,
                'class': cls
            })
    
    count = len(detections)
    
    # Draw results
    vis = img.copy()
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        conf = det['confidence']
        
        # Green box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (50, 220, 100), 2)
        
        # Label
        label = f"#{i} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), (50, 220, 100), -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), font, 0.5, (0, 0, 0), 1)
    
    # HUD
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w, 45), (15, 23, 42), -1)
    cv2.putText(vis, f"Cartons Detected: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (56, 189, 248), 2)
    cv2.putText(vis, f"Confidence: {confidence}", (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1)
    
    # Save output
    output_path = 'test_result.jpg'
    cv2.imwrite(output_path, vis)
    print(f"\nResult saved: {output_path}")
    
    # Print details
    print(f"\n{'='*40}")
    print(f"RESULTS:")
    print(f"{'='*40}")
    print(f"Cartons Detected: {count}")
    print(f"Confidence Threshold: {confidence}")
    print(f"{'='*40}")
    
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        w_box = x2 - x1
        h_box = y2 - y1
        print(f"  Carton #{i}: conf={det['confidence']:.2f}, size={w_box}x{h_box}")
    
    print(f"{'='*40}")
    
    return count, detections


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_detect.py <image_path> [model_path] [confidence]")
        print("Example: python test_detect.py test.jpg")
        print("Example: python test_detect.py test.jpg best.pt 0.5")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'best.pt'
    confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.36
    
    test_detection(image_path, model_path, confidence)
