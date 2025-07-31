#!/usr/bin/env python3
"""
Basic API test script to validate the translation API is working.
Run this after starting the server to test basic functionality.
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data.get('status', 'unknown')}")
            print(f"📊 Available models: {list(data.get('available_models', {}).keys())}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_models():
    """Test the models endpoint."""
    print("\n🤖 Testing models endpoint...")
    try:
        response = requests.get(f"{API_BASE}/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', {})
            print(f"✅ Models endpoint working. Found {len(models)} models:")
            for model_name, info in models.items():
                print(f"  • {model_name}: {info.get('description', 'No description')}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_translation():
    """Test a simple translation."""
    print("\n🙏 Testing translation endpoint...")
    try:
        payload = {
            "text": "OM MANI PADME HUM",
            "target_language": "English",
            "model_name": "claude",
            "text_type": "mantra"
        }
        
        response = requests.post(f"{API_BASE}/translate/single", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('results'):
                translation = data['results'][0]['translated_text']
                print(f"✅ Translation successful!")
                print(f"📝 Original: {payload['text']}")
                print(f"🔤 Translation: {translation}")
                return True
            else:
                print(f"❌ Translation failed: {data}")
                return False
        else:
            print(f"❌ Translation request failed: {response.status_code}")
            if response.text:
                print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Tibetan Buddhist Translation API - Basic Tests")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    tests_passed = 0
    total_tests = 3
    
    if test_health():
        tests_passed += 1
    
    if test_models():
        tests_passed += 1
    
    if test_translation():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The API is working correctly.")
        print("\n🚀 Next steps:")
        print("• Visit http://localhost:8000/docs for interactive API documentation")
        print("• Run python example_usage.py for more comprehensive examples")
        print("• Try translating your own Buddhist texts!")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")
        print("Make sure the server is running and API keys are configured.")

if __name__ == "__main__":
    main()