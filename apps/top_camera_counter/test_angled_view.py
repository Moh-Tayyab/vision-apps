"""Test Angled View Counter.

Usage:
    python test_angled_view.py <image_path>
    python test_angled_view.py dataset/images/Gemini_Generated_Image_yrtrlkyrtrlkyrtr.jpeg
    python test_angled_view.py test.jpg best.pt 0.36
"""

import sys
import cv2
from pathlib import Path
from ultralytics import YOLO
from angle_view_counter import analyze_angled_view, annotate_angled_view


def test_angled_view(
    image_path: str,
    model_path: str = 'best.pt',
    confidence: float = 0.36,
    overlap_threshold: float = 0.30,
):
    """Test angled view counting on a single image."""
    
    # Check model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return None
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image: {image_path}")
        return None
    
    print(f"Image loaded: {image_path} ({img.shape[1]}x{img.shape[0]})")
    
    # Run analysis
    result = analyze_angled_view(img, model, confidence=confidence, overlap_threshold=overlap_threshold)
    
    # Print results
    print("\n" + "="*60)
    print("ANGLED VIEW PALLET ANALYSIS RESULTS")
    print("="*60)
    print(f"Total Rows / Layers (Side View):  {result.total_rows}")
    print(f"Top Row Cartons (Top View):       {result.top_row_cartons}")
    print(f"Total Estimated Cartons:          {result.estimated_total_cartons}")
    print(f"Calculation Formula:              {result.top_row_cartons} (Top) x {result.total_rows} (Rows) = {result.estimated_total_cartons} Cartons")
    print(f"Total Visible Cartons Detected:   {result.total_cartons_detected}")
    print(f"Vertical Columns / Stacks:        {len(result.columns)}")
    print(f"Inference Time:                   {result.inference_time_ms:.1f} ms")
    print("="*60)
    
    # Print column details
    print("\nColumn Details:")
    for col in result.columns:
        print(f"  Column {col['column_index']}: {col['cartons_count']} cartons")
    
    # Annotate and save
    vis = annotate_angled_view(img, result)
    output_path = 'angled_view_result.jpg'
    cv2.imwrite(output_path, vis)
    print(f"\nResult saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_angled_view.py <image_path> [model_path] [confidence] [overlap_threshold]")
        print("Example: python test_angled_view.py dataset/images/Gemini_Generated_Image_yrtrlkyrtrlkyrtr.jpeg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'best.pt'
    confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.36
    overlap = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
    
    test_angled_view(image_path, model_path, confidence, overlap)
