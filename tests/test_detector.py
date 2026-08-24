"""
Test script for Carton Counter API.

Usage:
    python tests/test_detector.py [image1.jpg image2.jpg image3.jpg]

Env vars:
    CARTON_COUNTER_URL  base URL of the API (default http://localhost:8001)
    TEST_IMAGES_DIR     dir containing carton .jpg images (default ./test-images)
"""

import os
import sys

import requests

BASE_URL = os.getenv("CARTON_COUNTER_URL", "http://localhost:8001")


def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    return response.status_code == 200


def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/model/info")
    print("Model Info:", response.json())
    return response.status_code == 200


def test_detect(image_path: str):
    """Test single image detection (confidence is a query param)"""
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        response = requests.post(
            f"{BASE_URL}/detect", files=files, params={"confidence": 0.5}
        )

    result = response.json()
    print(f"\nDetection Result for {image_path}:")
    print(f"  Count: {result.get('count', 'N/A')}")
    print(f"  Detections: {len(result.get('detections', []))}")
    return response.status_code == 200


def test_count_multi_angle(image_paths: list):
    """Test multi-angle counting (API expects front/side/top file fields)"""
    files = []
    fields = ["front", "side", "top"]
    handles = []
    for field, path in zip(fields, image_paths):
        fh = open(path, "rb")
        handles.append(fh)
        files.append((field, (path, fh, "image/jpeg")))

    try:
        response = requests.post(f"{BASE_URL}/count", files=files)
    finally:
        for fh in handles:
            fh.close()

    result = response.json()
    print("\nMulti-Angle Count Result:")
    print(f"  Final Count: {result.get('count', 'N/A')}")
    print(f"  Per-View Counts: {result.get('per_view_counts', [])}")
    return response.status_code == 200


if __name__ == "__main__":
    print("=" * 50)
    print("Carton Counter API Tests")
    print("=" * 50)

    print("\n1. Testing Health Endpoint...")
    ok = test_health()

    print("\n2. Testing Model Info...")
    ok &= test_model_info()

    # Test detection with sample images
    test_images_dir = os.getenv("TEST_IMAGES_DIR", "test-images")
    if os.path.isdir(test_images_dir):
        images = sorted(
            os.path.join(test_images_dir, f)
            for f in os.listdir(test_images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )[:3]
        if images:
            print(f"\n3. Testing Detection with {images[0]}...")
            ok &= test_detect(images[0])
            if len(images) == 3:
                print("\n4. Testing Multi-Angle Count...")
                ok &= test_count_multi_angle(images)
        else:
            print("\n3. No test images found")
    else:
        print(f"\n3. Test images directory not found: {test_images_dir}")

    print("\n" + "=" * 50)
    print("Tests Complete!", "ALL PASSED" if ok else "SOME FAILED")
    print("=" * 50)
    sys.exit(0 if ok else 1)
