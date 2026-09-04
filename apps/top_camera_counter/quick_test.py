"""
Quick Test - Image Upload
Run karo aur image select karo, result dikhayega
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tkinter import Tk, filedialog


def quick_test():
    """Upload image and test detection."""
    
    # Check model
    if not Path('best.pt').exists():
        print("ERROR: best.pt not found!")
        print("Model train karke root mein rakho")
        return
    
    # Select image
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    
    if not file_path:
        print("No image selected")
        return
    
    # Load model
    model = YOLO('best.pt')
    
    # Run detection
    results = model(file_path, conf=0.36, verbose=False)
    
    # Count
    count = len(results[0].boxes) if results[0].boxes else 0
    
    # Annotate
    annotated = results[0].plot()
    
    # Save
    cv2.imwrite('result.jpg', annotated)
    
    # Show
    print(f"\nDetected: {count} cartons")
    print(f"Result saved: result.jpg")
    
    # Display image
    cv2.imshow('Detection Result', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    quick_test()
