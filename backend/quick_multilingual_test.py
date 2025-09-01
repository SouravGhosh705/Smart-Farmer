#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json

def test_multilingual_prediction():
    """Test multilingual crop prediction"""
    
    print("🧪 Testing Multilingual Crop Prediction")
    print("=" * 50)
    
    # Test data
    test_request = {
        "N": 90,
        "P": 42,
        "K": 43,
        "ph": 6.5,
        "rainfall": 200,
        "state": "gujarat",
        "city": "ahmedabad",
        "language": "hindi"
    }
    
    try:
        print("📡 Sending request to crop_prediction endpoint...")
        print(f"   Request: {json.dumps(test_request, indent=2)}")
        
        response = requests.post("http://localhost:8000/crop_prediction", 
                               json=test_request, 
                               timeout=10)
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS! Multilingual prediction working!")
            print(f"\n🎯 Results:")
            print(f"   Top Crop: {data.get('crop', 'N/A')}")
            print(f"   Language: {data.get('language', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            
            if 'crop_list' in data:
                print(f"\n🌾 Top 5 Recommendations:")
                for i, (crop, conf) in enumerate(data['crop_list'][:5], 1):
                    print(f"      {i}. {crop}: {conf*100:.1f}%")
            
            if 'supported_languages' in data:
                print(f"\n🌍 Supported Languages: {len(data['supported_languages'])} languages")
            
            return True
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    test_multilingual_prediction()
