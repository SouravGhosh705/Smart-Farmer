#!/usr/bin/env python
# -*- coding: utf-8 -*-

def test_model_directly():
    """Test the crop recommendation model directly without HTTP"""
    
    try:
        from joblib import load
        import numpy as np
        import os
        
        print("🧪 Testing Crop Recommendation Model Directly...")
        print("=" * 60)
        
        # Check if model exists
        model_path = 'static/models/crop_recommendation_model.joblib'
        
        if not os.path.exists(model_path):
            print("❌ Model file not found!")
            return False
        
        print("✅ Model file exists")
        
        # Load the model
        model = load(model_path)
        print("✅ Model loaded successfully")
        
        # Test input (same as frontend would send)
        test_input = np.array([[50, 25, 30, 28, 70, 6.5, 200]])  # N, P, K, temp, humidity, pH, rainfall
        
        # Make prediction
        prediction = model.predict(test_input)
        probabilities = model.predict_proba(test_input)
        classes = model.classes_
        
        print("✅ Prediction successful!")
        print(f"🏆 Top recommended crop: {prediction[0]}")
        print("🌾 Top 5 crops with probabilities:")
        
        # Create crop list with probabilities
        crop_probs = [(classes[i], probabilities[0][i]) for i in range(len(classes))]
        crop_probs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (crop, prob) in enumerate(crop_probs[:5], 1):
            confidence = round(prob * 100, 2)
            print(f"   {i}. {crop.title()}: {confidence}%")
        
        print("=" * 60)
        print("🎉 CROP RECOMMENDATION MODEL IS WORKING PERFECTLY!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        return False

def check_dataset():
    """Check if dataset is properly loaded"""
    try:
        import pandas as pd
        import os
        
        print("📊 Checking Dataset...")
        
        dataset_path = 'static/datasets/crop_recommendation.csv'
        
        if not os.path.exists(dataset_path):
            print("❌ Dataset file not found!")
            return False
        
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded: {len(df)} records")
        print(f"✅ Columns: {list(df.columns)}")
        print(f"✅ Unique crops: {df['label'].nunique()}")
        print(f"✅ Crop types: {sorted(df['label'].unique())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking dataset: {str(e)}")
        return False

if __name__ == "__main__":
    print("🌾 SMART FARMER - DIRECT MODEL TEST")
    print("=" * 80)
    
    # Check dataset
    dataset_ok = check_dataset()
    print()
    
    if dataset_ok:
        # Test model directly
        model_ok = test_model_directly()
        
        if model_ok:
            print("\n✅ ALL TESTS PASSED!")
            print("   • Dataset is loaded correctly")
            print("   • Model is trained and working")
            print("   • Predictions are being generated")
            print("\n🔧 NEXT STEPS:")
            print("   1. Start the backend server: python app.py")
            print("   2. Start the frontend server")
            print("   3. Test the crop recommendation feature")
        else:
            print("\n❌ Model test failed!")
    else:
        print("\n❌ Dataset test failed!")
    
    print("=" * 80)
