"""Simplified Carton Counter - Angled View.

Counts total rows (side view) and top cartons (top view) from single angled picture.
Usage: python simple_counter.py <image_path>
"""

import sys
import cv2
from pathlib import Path
from ultralytics import YOLO
from angle_view_counter import analyze_angled_view, annotate_angled_view


def count_cartons(
    image_path: str,
    model_path: str = 'best.pt',
    confidence: float = 0.36,
    overlap_threshold: float = 0.30,
) -> dict:
    """Count rows and top cartons from angled view image."""
    if not Path(model_path).exists():
        return {"error": f"Model not found: {model_path}"}
    
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    
    if img is None:
        return {"error": f"Could not read image: {image_path}"}
    
    result = analyze_angled_view(img, model, confidence=confidence, overlap_threshold=overlap_threshold)
    return result.to_dict()


def annotate_image(
    image_path: str,
    output_path: str = 'count_result.jpg',
    model_path: str = 'best.pt',
    confidence: float = 0.36,
    overlap_threshold: float = 0.30,
) -> str:
    """Annotate image with detection results."""
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    
    if img is None:
        return ""
    
    result = analyze_angled_view(img, model, confidence=confidence, overlap_threshold=overlap_threshold)
    vis = annotate_angled_view(img, result)
    cv2.imwrite(output_path, vis)
    
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_counter.py <image_path> [model_path] [confidence]")
        print("Example: python simple_counter.py dataset/images/Gemini_Generated_Image_yrtrlkyrtrlkyrtr.jpeg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'best.pt'
    confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.36
    
    result = count_cartons(image_path, model_path, confidence)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("CARTON COUNTING RESULTS")
    print("="*50)
    print(f"Total Rows/Layers (Side View):  {result['total_rows']}")
    print(f"Top Row Cartons (Top View):     {result['top_row_cartons']}")
    print(f"Total Estimated Cartons:        {result['estimated_total_cartons']}")
    print(f"Formula:                        {result['formula']}")
    print(f"Total Visible Detected:         {result['total_cartons_detected']}")
    print(f"Vertical Columns:               {result['columns_count']}")
    print("="*50)
    
    output = annotate_image(image_path, model_path=model_path, confidence=confidence)
    if output:
        print(f"\nResult saved: {output}")
