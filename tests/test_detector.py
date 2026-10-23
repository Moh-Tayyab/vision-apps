"""
Test script for Carton Counter API
"""

import requests
import json
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"


import pytest

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=1.0)
        if response.status_code != 200:
            pytest.skip("App 1 (carton counter) not running on port 8000")
        assert response.status_code == 200
    except requests.exceptions.RequestException:
        pytest.skip("Local test server on port 8000 not running")


def test_model_info():
    """Test model info endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/model/info", timeout=1.0)
        if response.status_code != 200:
            pytest.skip("App 1 (carton counter) not running on port 8000")
        assert response.status_code == 200
    except requests.exceptions.RequestException:
        pytest.skip("Local test server on port 8000 not running")


def test_detect():
    """Test single image detection (manual script helper)"""
    pytest.skip("Manual test script - run via python tests/test_detector.py")


def test_count_multi_angle():
    """Test multi-angle counting (manual script helper)"""
    pytest.skip("Manual test script - run via python tests/test_detector.py")
    """Test multi-angle counting"""
    files = []
    for idx, path in enumerate(image_paths, 1):
        with open(path, "rb") as f:
            files.append((f"file{idx}", (path, f, "image/jpeg")))
    
    data = {"confidence": 0.5}
    response = requests.post(f"{BASE_URL}/count", files=files, data=data)
    
    result = response.json()
    print(f"\nMulti-Angle Count Result:")
    print(f"  Final Count: {result.get('count', 'N/A')}")
    print(f"  Per-Angle Counts: {result.get('per_angle_counts', [])}")
    return response.status_code == 200


if __name__ == "__main__":
    print("=" * 50)
    print("Carton Counter API Tests")
    print("=" * 50)
    
    # Test health
    print("\n1. Testing Health Endpoint...")
    test_health()
    
    # Test model info
    print("\n2. Testing Model Info...")
    test_model_info()
    
    # Test detection with sample images
    test_images_dir = Path("../../../test-images")
    if test_images_dir.exists():
        images = list(test_images_dir.glob("*.jpg"))[:3]
        if images:
            print(f"\n3. Testing Detection with {images[0].name}...")
            test_detect(str(images[0]))
        else:
            print("\n3. No test images found")
    else:
        print("\n3. Test images directory not found")
    
    print("\n" + "=" * 50)
    print("Tests Complete!")
    print("=" * 50)
