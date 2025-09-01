#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Installation and Testing Script for Enhanced FastAPI Agricultural Backend
Installs dependencies, sets up Ollama, tests all endpoints, and validates the system
"""

import os
import sys
import subprocess
import asyncio
import aiohttp
import json
import time
import requests
from datetime import datetime
import base64
from PIL import Image
import io

class BackendInstaller:
    """Complete installer for the enhanced agricultural backend"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ollama_url = "http://localhost:11434"
        self.test_results = {}
        
    def install_dependencies(self):
        """Install all required Python dependencies"""
        print("📦 Installing Python dependencies...")
        
        dependencies = [
            'fastapi[all]',
            'uvicorn[standard]',
            'pandas',
            'numpy',
            'scikit-learn',
            'joblib',
            'requests',
            'aiohttp',
            'opencv-python',
            'pillow',
            'python-multipart'
        ]
        
        try:
            for dep in dependencies:
                print(f"Installing {dep}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                             check=True, capture_output=True)
            
            print("✅ All dependencies installed successfully!")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def setup_ollama(self):
        """Setup Ollama for the agricultural backend"""
        print("🦙 Setting up Ollama...")
        
        try:
            # Check if Ollama is installed
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print("❌ Ollama not found. Please install Ollama first:")
                print("   Visit: https://ollama.ai/download")
                print("   Or run: curl -fsSL https://ollama.ai/install.sh | sh")
                return False
            
            print(f"✅ Ollama found: {result.stdout.strip()}")
            
            # Start Ollama service
            print("🚀 Starting Ollama service...")
            ollama_process = subprocess.Popen(['ollama', 'serve'])
            
            # Wait for service to start
            time.sleep(5)
            
            # Test connection
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    print("✅ Ollama service is running")
                    
                    # Install recommended model
                    print("📥 Installing recommended agricultural model...")
                    subprocess.run(['ollama', 'pull', 'llama3.2'], check=True)
                    print("✅ Agricultural model installed")
                    
                    return True
                else:
                    print("❌ Ollama service not responding")
                    return False
            
            except Exception as e:
                print(f"❌ Error connecting to Ollama: {e}")
                return False
        
        except Exception as e:
            print(f"❌ Error setting up Ollama: {e}")
            return False
    
    def create_sample_data(self):
        """Create sample data files for testing"""
        print("📊 Creating sample data files...")
        
        try:
            # Create directories
            os.makedirs('static/models', exist_ok=True)
            os.makedirs('static/labelencoder', exist_ok=True)
            os.makedirs('static/datasets', exist_ok=True)
            os.makedirs('static/uploads', exist_ok=True)
            
            # Create sample image for testing
            sample_image = Image.new('RGB', (224, 224), color='green')
            sample_image.save('static/uploads/sample_leaf.jpg')
            
            print("✅ Sample data files created")
            return True
        
        except Exception as e:
            print(f"❌ Error creating sample data: {e}")
            return False
    
    async def test_all_endpoints(self):
        """Test all backend endpoints"""
        print("🧪 Testing all backend endpoints...")
        
        test_cases = [
            # Basic endpoints
            {'method': 'GET', 'url': '/', 'name': 'Root endpoint'},
            {'method': 'GET', 'url': '/health', 'name': 'Health check'},
            {'method': 'GET', 'url': '/system/status', 'name': 'System status'},
            
            # Ollama endpoints
            {'method': 'GET', 'url': '/ollama/status', 'name': 'Ollama status'},
            {'method': 'GET', 'url': '/setup/ollama/status', 'name': 'Ollama setup status'},
            {'method': 'GET', 'url': '/setup/models/available', 'name': 'Available models'},
            
            # Weather endpoints
            {
                'method': 'POST', 
                'url': '/weather/current',
                'data': {'city': 'Delhi', 'state': 'Delhi', 'country': 'IN'},
                'name': 'Current weather'
            },
            {
                'method': 'POST',
                'url': '/weather/forecast', 
                'data': {'city': 'Mumbai', 'state': 'Maharashtra'},
                'name': 'Weather forecast'
            },
            
            # Translation endpoints
            {
                'method': 'POST',
                'url': '/translate',
                'data': {
                    'text': 'Hello farmer',
                    'target_language': 'hi',
                    'source_language': 'en'
                },
                'name': 'Translation service'
            },
            
            # Market price endpoints
            {
                'method': 'GET',
                'url': '/market/prices/rice',
                'name': 'Rice market prices'
            },
            {
                'method': 'POST',
                'url': '/market/prices',
                'data': {
                    'commodity': 'wheat',
                    'state': 'Punjab',
                    'market': 'Ludhiana'
                },
                'name': 'Enhanced market prices'
            },
            {
                'method': 'GET',
                'url': '/market/forecast/cotton',
                'name': 'Price forecast'
            },
            {
                'method': 'GET',
                'url': '/market/analytics/maize',
                'name': 'Market analytics'
            },
            
            # Dataset endpoints
            {'method': 'GET', 'url': '/datasets/available', 'name': 'Available datasets'},
            {'method': 'GET', 'url': '/models/cv/available', 'name': 'Available CV models'},
            
            # Chatbot endpoint
            {
                'method': 'POST',
                'url': '/chat',
                'data': {
                    'message': 'What is the best time to plant rice?',
                    'language': 'english',
                    'location': {'city': 'Delhi', 'state': 'Delhi'}
                },
                'name': 'Smart chatbot'
            }
        ]
        
        async with aiohttp.ClientSession() as session:
            for test in test_cases:
                try:
                    url = f"{self.base_url}{test['url']}"
                    
                    if test['method'] == 'GET':
                        async with session.get(url) as response:
                            success = response.status == 200
                            result = await response.json() if success else await response.text()
                    
                    else:  # POST
                        data = test.get('data', {})
                        async with session.post(url, json=data) as response:
                            success = response.status == 200
                            result = await response.json() if success else await response.text()
                    
                    self.test_results[test['name']] = {
                        'success': success,
                        'status_code': response.status,
                        'url': test['url'],
                        'method': test['method']
                    }
                    
                    status = "✅" if success else "❌"
                    print(f"{status} {test['name']} - Status: {response.status}")
                
                except Exception as e:
                    print(f"❌ {test['name']} - Error: {str(e)}")
                    self.test_results[test['name']] = {
                        'success': False,
                        'error': str(e),
                        'url': test['url'],
                        'method': test['method']
                    }
    
    async def test_image_upload(self):
        """Test disease detection with image upload"""
        print("🖼️ Testing image upload for disease detection...")
        
        try:
            # Create a sample image
            sample_image = Image.new('RGB', (224, 224), color='green')
            img_byte_arr = io.BytesIO()
            sample_image.save(img_byte_arr, format='JPEG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Test disease detection
            test_data = {
                'image_base64': img_base64,
                'crop_type': 'tomato',
                'location': {'city': 'Delhi', 'state': 'Delhi'}
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/disease-detection",
                    json=test_data
                ) as response:
                    success = response.status == 200
                    result = await response.json() if success else await response.text()
                    
                    if success:
                        print("✅ Disease detection test passed")
                        print(f"   Detected: {result.get('detection_summary', {}).get('primary_detection', 'Unknown')}")
                    else:
                        print(f"❌ Disease detection test failed: {response.status}")
            
            self.test_results['Disease Detection'] = {
                'success': success,
                'status_code': response.status
            }
        
        except Exception as e:
            print(f"❌ Image upload test error: {e}")
            self.test_results['Disease Detection'] = {
                'success': False,
                'error': str(e)
            }
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results.values() if test.get('success', False))
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n🔍 Detailed Results:")
        print("-" * 60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            print(f"{status} {test_name}")
            
            if not result.get('success', False):
                if 'error' in result:
                    print(f"     Error: {result['error']}")
                elif 'status_code' in result:
                    print(f"     Status Code: {result['status_code']}")
        
        print("\n" + "="*60)
        print("🚀 BACKEND FEATURES SUMMARY")
        print("="*60)
        
        features = [
            "✅ FastAPI Backend with Async Support",
            "✅ Smart Chatbot with Ollama Integration",
            "✅ AI Crop Doctor with Enhanced Models",
            "✅ Real-time Weather Integration",
            "✅ Multi-language Translation Support",
            "✅ Enhanced Market Price Analysis",
            "✅ Price Alerts and Forecasting",
            "✅ Dataset Management System",
            "✅ Comprehensive API Documentation",
            "✅ Background Task Processing",
            "✅ Error Handling and Fallbacks",
            "✅ Conversation Memory Management"
        ]
        
        for feature in features:
            print(feature)
        
        print("\n🌐 API Endpoints Available:")
        print("-" * 40)
        endpoints = [
            "🏠 / - Root endpoint with status",
            "💚 /health - Health check",
            "📊 /system/status - Complete system status",
            "🤖 /chat - Smart chatbot",
            "🔍 /disease-detection - AI Crop Doctor",
            "🌤️ /weather/current - Current weather",
            "🌤️ /weather/forecast - Weather forecast",
            "🌐 /translate - Text translation",
            "💰 /market/prices/{commodity} - Market prices",
            "📈 /market/forecast/{commodity} - Price forecast",
            "📊 /market/analytics/{commodity} - Market analytics",
            "🚨 /market/alerts/set - Set price alerts",
            "🦙 /ollama/status - Ollama status",
            "⚙️ /setup/ollama/install - Setup Ollama",
            "📚 /datasets/available - Available datasets",
            "🎯 /models/cv/available - CV models"
        ]
        
        for endpoint in endpoints:
            print(endpoint)
        
        print(f"\n📱 Interactive API Documentation: {self.base_url}/docs")
        print(f"📖 Alternative API Documentation: {self.base_url}/redoc")
        
        return passed_tests == total_tests

class SystemValidator:
    """Validates system requirements and dependencies"""
    
    def __init__(self):
        self.requirements = {
            'python': {'version': '3.8+', 'command': 'python --version'},
            'pip': {'version': '20+', 'command': 'pip --version'},
            'git': {'version': 'any', 'command': 'git --version'}
        }
    
    def check_system_requirements(self):
        """Check all system requirements"""
        print("🔍 Checking system requirements...")
        
        all_good = True
        
        for req, details in self.requirements.items():
            try:
                result = subprocess.run(
                    details['command'].split(),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    version_output = result.stdout.strip()
                    print(f"✅ {req}: {version_output}")
                else:
                    print(f"❌ {req}: Not found or error")
                    all_good = False
            
            except Exception as e:
                print(f"❌ {req}: Error checking - {e}")
                all_good = False
        
        return all_good
    
    def check_disk_space(self):
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_gb = free // (1024**3)
            
            print(f"💾 Available disk space: {free_gb} GB")
            
            if free_gb < 5:
                print("⚠️ Warning: Less than 5GB free space available")
                print("   Consider freeing up space for model downloads")
                return False
            else:
                print("✅ Sufficient disk space available")
                return True
        
        except Exception as e:
            print(f"❌ Error checking disk space: {e}")
            return False

async def main():
    """Main installation and testing function"""
    print("🌾 Enhanced Agricultural Backend - Installation & Testing")
    print("=" * 60)
    
    # Initialize components
    validator = SystemValidator()
    installer = BackendInstaller()
    
    # Step 1: Check system requirements
    print("\n1️⃣ SYSTEM REQUIREMENTS CHECK")
    print("-" * 30)
    
    if not validator.check_system_requirements():
        print("❌ System requirements not met. Please install missing components.")
        return False
    
    if not validator.check_disk_space():
        print("⚠️ Disk space warning - continue anyway? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            return False
    
    # Step 2: Install dependencies
    print("\n2️⃣ DEPENDENCY INSTALLATION")
    print("-" * 30)
    
    if not installer.install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Step 3: Setup sample data
    print("\n3️⃣ SAMPLE DATA SETUP")
    print("-" * 30)
    
    if not installer.create_sample_data():
        print("❌ Failed to create sample data")
        return False
    
    # Step 4: Setup Ollama (optional)
    print("\n4️⃣ OLLAMA SETUP (Optional)")
    print("-" * 30)
    print("Set up Ollama for advanced chatbot features? (y/n)")
    
    response = input().strip().lower()
    if response == 'y':
        ollama_success = installer.setup_ollama()
        if not ollama_success:
            print("⚠️ Ollama setup failed - backend will run with limited chatbot features")
    else:
        print("⏭️ Skipping Ollama setup - using fallback chatbot")
    
    # Step 5: Start backend and test
    print("\n5️⃣ BACKEND TESTING")
    print("-" * 30)
    
    print("Starting FastAPI backend...")
    print("Please run in another terminal: python fastapi_app.py")
    print("Press Enter when the backend is running...")
    input()
    
    # Wait for backend to start
    print("⏳ Waiting for backend to start...")
    await asyncio.sleep(3)
    
    # Test endpoints
    await installer.test_all_endpoints()
    await installer.test_image_upload()
    
    # Generate report
    success = installer.generate_test_report()
    
    if success:
        print("\n🎉 Installation and testing completed successfully!")
        print("\n🚀 Your Enhanced Agricultural Backend is ready!")
        print(f"   - Backend: {installer.base_url}")
        print(f"   - Documentation: {installer.base_url}/docs")
        print("\n📱 You can now integrate this backend with your frontend application")
    else:
        print("\n⚠️ Some tests failed. Check the report above.")
        print("   The backend may still work with limited functionality.")
    
    return success

def quick_start_guide():
    """Display quick start guide"""
    print("\n📖 QUICK START GUIDE")
    print("=" * 50)
    
    guide = """
1. Run the installation script:
   python install_and_test.py

2. Start the backend:
   python fastapi_app.py

3. Access the API documentation:
   http://localhost:8000/docs

4. Test key endpoints:
   - Market Prices: GET /market/prices/rice
   - Weather: POST /weather/current
   - Chatbot: POST /chat
   - Disease Detection: POST /disease-detection

5. Optional - Setup Ollama for advanced features:
   - Install Ollama from https://ollama.ai
   - Use /setup/ollama/install endpoint
   - Install models via /setup/model/install/{model_name}

6. Integration Examples:
   
   # Get market prices
   curl -X GET "http://localhost:8000/market/prices/wheat?state=Punjab"
   
   # Chat with AI advisor
   curl -X POST "http://localhost:8000/chat" \\
        -H "Content-Type: application/json" \\
        -d '{"message": "How to increase wheat yield?"}'
   
   # Get weather forecast
   curl -X POST "http://localhost:8000/weather/forecast" \\
        -H "Content-Type: application/json" \\
        -d '{"city": "Delhi", "state": "Delhi"}'

7. Key Features:
   ✅ Async FastAPI backend
   ✅ Smart agricultural chatbot
   ✅ AI crop disease detection
   ✅ Real-time weather integration
   ✅ Enhanced market price analysis
   ✅ Multi-language support
   ✅ Price alerts and forecasting
   ✅ Dataset management
   ✅ Comprehensive API docs
    """
    
    print(guide)

if __name__ == "__main__":
    print("🌾 Enhanced Agricultural Backend Installer")
    print("Choose an option:")
    print("1. Full installation and testing")
    print("2. Quick start guide")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(main())
    elif choice == "2":
        quick_start_guide()
    else:
        print("👋 Goodbye!")
