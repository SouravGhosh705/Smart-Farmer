#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced FastAPI Backend for AI-Assisted Farming Application
Integrates Smart Chatbot and AI Crop Doctor with free online services
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from joblib import dump, load
import os
import logging
import traceback
from datetime import datetime, date, timedelta
import requests
import json
import uuid
import re
import asyncio
import aiohttp
import base64
import cv2
from PIL import Image
import io
from contextlib import asynccontextmanager

# Import existing modules
from pest_disease_detection import plant_detector, detect_disease_from_image, decode_base64_to_image
from multilingual_system import get_multilingual_system
from enhanced_crop_doctor import enhanced_detector, detect_disease_enhanced, train_disease_detection_model, list_available_models as list_cv_models
from ollama_setup import setup_ollama_for_agriculture, install_agricultural_model, get_ollama_status, OllamaSetupAssistant, OllamaManager
from enhanced_market_prices import (
    get_enhanced_market_prices, 
    get_price_forecast, 
    get_market_analytics,
    market_aggregator,
    price_alert_system
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class CropPredictionRequest(BaseModel):
    N: float = Field(..., description="Nitrogen content")
    P: float = Field(..., description="Phosphorus content") 
    K: float = Field(..., description="Potassium content")
    ph: float = Field(..., description="Soil pH")
    rainfall: float = Field(..., description="Rainfall in mm")
    state: str = Field(..., description="State name")
    city: str = Field(..., description="City name")
    language: Optional[str] = Field("english", description="Response language")

class YieldPredictionRequest(BaseModel):
    state: str
    city: str
    season: str
    crop: str
    area: float

class ChatbotRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_id: Optional[str] = Field(None, description="User ID for conversation tracking")
    language: Optional[str] = Field("english", description="Language for response")
    location: Optional[Dict[str, str]] = Field(None, description="User location data")

class DiseaseDetectionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded plant image")
    crop_type: Optional[str] = Field(None, description="Type of crop")
    location: Optional[Dict[str, str]] = Field(None, description="Location data")

class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language code")
    source_language: Optional[str] = Field("auto", description="Source language code")

class WeatherRequest(BaseModel):
    city: str
    state: Optional[str] = None
    country: str = Field("IN", description="Country code")

class MarketPriceRequest(BaseModel):
    commodity: str = Field(..., description="Commodity name")
    state: Optional[str] = Field(None, description="State name")
    market: Optional[str] = Field(None, description="Market/city name")

class PriceAlertRequest(BaseModel):
    commodity: str = Field(..., description="Commodity name")
    target_price: float = Field(..., description="Target price in Rs/Quintal")
    alert_type: str = Field("above", description="Alert type: 'above' or 'below'")

# Global variables for ML models and services
global_models = {}
global_encoders = {}
conversation_memory = {}

class OllamaService:
    """Service for interacting with local Ollama LLM"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.2"  # Default model
    
    async def check_ollama_status(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Ollama not accessible: {str(e)}")
            return False
    
    async def generate_response(self, prompt: str, context: Dict = None) -> str:
        """Generate response using Ollama"""
        try:
            # Build the full prompt with agricultural context
            system_prompt = """You are an expert agricultural advisor and crop consultant. 
            You provide helpful, accurate, and practical farming advice. 
            Focus on sustainable farming practices, crop management, and agricultural best practices.
            Be concise but comprehensive in your responses."""
            
            if context:
                system_prompt += f"\nContext: {json.dumps(context, indent=2)}"
            
            full_prompt = f"{system_prompt}\n\nUser Question: {prompt}\n\nResponse:"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "Sorry, I couldn't generate a response.")
                    else:
                        logger.error(f"Ollama API error: {response.status}")
                        return "I'm having trouble accessing the AI model. Please try again."
        
        except Exception as e:
            logger.error(f"Error generating Ollama response: {str(e)}")
            return "I'm currently experiencing technical difficulties. Please try again."

class WeatherService:
    """Service for getting weather data"""
    
    def __init__(self):
        self.api_key = "ff049be539ac8642b805155154206e4c"  # Free OpenWeatherMap API key
        self.base_url = "http://api.openweathermap.org/data/2.5"
    
    async def get_current_weather(self, city: str, state: str = None, country: str = "IN") -> Dict:
        """Get current weather data"""
        try:
            # Construct location query
            if state:
                location = f"{city},{state},{country}"
            else:
                location = f"{city},{country}"
            
            url = f"{self.base_url}/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'temperature': float(data['main']['temp']),
                            'humidity': float(data['main']['humidity']),
                            'pressure': float(data['main']['pressure']),
                            'weather_main': data['weather'][0]['main'],
                            'weather_description': data['weather'][0]['description'],
                            'wind_speed': float(data.get('wind', {}).get('speed', 0)),
                            'location': data['name'],
                            'country': data['sys']['country'],
                            'feels_like': float(data['main']['feels_like']),
                            'visibility': data.get('visibility', 10000),
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        raise Exception(f"Weather API error: {response.status}")
        
        except Exception as e:
            logger.warning(f"Error fetching weather data: {str(e)}")
            # Return fallback data
            return {
                'temperature': 28.0,
                'humidity': 70.0,
                'pressure': 1013.25,
                'weather_main': 'Clear',
                'weather_description': 'clear sky',
                'wind_speed': 5.0,
                'location': city,
                'country': country,
                'feels_like': 30.0,
                'visibility': 10000,
                'timestamp': datetime.now().isoformat(),
                'note': 'Fallback data - actual weather unavailable'
            }

    async def get_weather_forecast(self, city: str, state: str = None, country: str = "IN", days: int = 5) -> Dict:
        """Get weather forecast"""
        try:
            if state:
                location = f"{city},{state},{country}"
            else:
                location = f"{city},{country}"
            
            url = f"{self.base_url}/forecast"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric",
                "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        forecasts = []
                        
                        for item in data['list']:
                            forecasts.append({
                                'datetime': item['dt_txt'],
                                'temperature': float(item['main']['temp']),
                                'humidity': float(item['main']['humidity']),
                                'weather': item['weather'][0]['description'],
                                'wind_speed': float(item.get('wind', {}).get('speed', 0)),
                                'precipitation_prob': float(item.get('pop', 0)) * 100
                            })
                        
                        return {
                            'location': data['city']['name'],
                            'country': data['city']['country'],
                            'forecasts': forecasts,
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        raise Exception(f"Forecast API error: {response.status}")
        
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {str(e)}")
            return {'error': str(e)}

class TranslationService:
    """Service for text translation using free APIs"""
    
    def __init__(self):
        self.mymemory_url = "https://api.mymemory.translated.net/get"
        self.libre_translate_url = "https://libretranslate.de/translate"  # Free instance
    
    async def translate_text(self, text: str, target_lang: str, source_lang: str = "auto") -> Dict:
        """Translate text using free translation services"""
        try:
            # Try LibreTranslate first
            result = await self._translate_with_libretranslate(text, target_lang, source_lang)
            if result.get('translated_text'):
                return result
            
            # Fallback to MyMemory
            result = await self._translate_with_mymemory(text, target_lang, source_lang)
            return result
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return {
                'translated_text': text,  # Return original text as fallback
                'source_language': source_lang,
                'target_language': target_lang,
                'error': str(e)
            }
    
    async def _translate_with_libretranslate(self, text: str, target_lang: str, source_lang: str) -> Dict:
        """Translate using LibreTranslate"""
        try:
            payload = {
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.libre_translate_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'translated_text': data.get('translatedText', text),
                            'source_language': source_lang,
                            'target_language': target_lang,
                            'service': 'LibreTranslate'
                        }
        except Exception as e:
            logger.warning(f"LibreTranslate failed: {str(e)}")
            return {}
    
    async def _translate_with_mymemory(self, text: str, target_lang: str, source_lang: str) -> Dict:
        """Translate using MyMemory"""
        try:
            params = {
                "q": text,
                "langpair": f"{source_lang}|{target_lang}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.mymemory_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('responseStatus') == 200:
                            return {
                                'translated_text': data['responseData']['translatedText'],
                                'source_language': source_lang,
                                'target_language': target_lang,
                                'service': 'MyMemory',
                                'match_quality': data['responseData'].get('match', 0)
                            }
        except Exception as e:
            logger.warning(f"MyMemory translation failed: {str(e)}")
            return {}

class DatasetManager:
    """Manager for downloading and handling free agricultural datasets"""
    
    def __init__(self):
        self.dataset_urls = {
            'plantvillage': {
                'url': 'https://github.com/spMohanty/PlantVillage-Dataset/archive/master.zip',
                'description': 'PlantVillage dataset for plant disease classification',
                'size': '~1.5GB',
                'classes': 38
            },
            'plantdoc': {
                'url': 'https://github.com/pratikkayal/PlantDoc-Dataset/archive/master.zip', 
                'description': 'PlantDoc dataset for plant disease detection and classification',
                'size': '~2GB',
                'classes': 27
            }
        }
    
    async def download_dataset(self, dataset_name: str, download_path: str) -> Dict:
        """Download a dataset"""
        try:
            if dataset_name not in self.dataset_urls:
                return {'error': f'Dataset {dataset_name} not available'}
            
            dataset_info = self.dataset_urls[dataset_name]
            url = dataset_info['url']
            
            # Create download directory
            os.makedirs(download_path, exist_ok=True)
            
            # Download with progress tracking (simplified)
            filename = f"{dataset_name}.zip"
            filepath = os.path.join(download_path, filename)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(filepath, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        
                        return {
                            'status': 'success',
                            'dataset': dataset_name,
                            'filepath': filepath,
                            'size_mb': os.path.getsize(filepath) / (1024 * 1024),
                            'info': dataset_info
                        }
                    else:
                        return {'error': f'Download failed with status {response.status}'}
        
        except Exception as e:
            logger.error(f"Dataset download error: {str(e)}")
            return {'error': str(e)}

    async def list_available_datasets(self) -> Dict:
        """List all available free datasets"""
        return {
            'available_datasets': self.dataset_urls,
            'total_datasets': len(self.dataset_urls)
        }

# Services initialization
ollama_service = OllamaService()
weather_service = WeatherService()
translation_service = TranslationService()
dataset_manager = DatasetManager()

# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Smart Farmer FastAPI Backend")
    
    # Create necessary directories
    os.makedirs('static/models', exist_ok=True)
    os.makedirs('static/labelencoder', exist_ok=True)
    os.makedirs('static/datasets', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    # Check Ollama status
    ollama_status = await ollama_service.check_ollama_status()
    if ollama_status:
        logger.info("✅ Ollama service is running")
    else:
        logger.warning("⚠️ Ollama service not available - chatbot will use fallback responses")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Smart Farmer API")

app = FastAPI(
    title="Smart Farmer AI API",
    description="Enhanced AI-Assisted Farming Application with Smart Chatbot and AI Crop Doctor",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service status"""
    ollama_status = await ollama_service.check_ollama_status()
    
    return {
        "message": "Smart Farmer AI API - Enhanced Version",
        "version": "3.0.0",
        "status": "running",
        "services": {
            "chatbot": ollama_status,
            "weather": True,
            "translation": True,
            "disease_detection": True,
            "crop_recommendation": True,
            "yield_prediction": True
        },
        "timestamp": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    model_status = {
        'crop_recommendation': os.path.exists('static/models/crop_recommendation_model.joblib'),
        'yield_prediction': os.path.exists('static/models/yield_prediction_model.joblib'),
        'label_encoders': all(os.path.exists(f'static/labelencoder/{name}_le.joblib') 
                            for name in ['statename', 'districtname', 'season', 'crop'])
    }
    
    service_status = {
        'ollama': await ollama_service.check_ollama_status(),
        'weather_api': True,  # Assume available unless we test it
        'translation_api': True
    }
    
    return {
        "status": "healthy",
        "models": model_status,
        "services": service_status,
        "timestamp": datetime.now().isoformat()
    }

# Smart Chatbot endpoints
@app.post("/chat")
async def smart_chatbot(request: ChatbotRequest):
    """Smart chatbot with Ollama integration and agricultural knowledge"""
    try:
        user_id = request.user_id or str(uuid.uuid4())
        message = request.message
        language = request.language
        location = request.location or {}
        
        # Initialize conversation memory for user
        if user_id not in conversation_memory:
            conversation_memory[user_id] = {
                'messages': [],
                'context': location,
                'language': language,
                'created_at': datetime.now().isoformat()
            }
        
        # Add user message to memory
        conversation_memory[user_id]['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Detect intent and extract entities using existing engine
        from app import advisory_engine
        intent = advisory_engine.detect_intent(message)
        entities = advisory_engine.extract_entities(message)
        
        # Build context for AI response
        context = {
            'user_location': location,
            'conversation_history': conversation_memory[user_id]['messages'][-5:],  # Last 5 messages
            'detected_intent': intent,
            'extracted_entities': entities,
            'user_language': language
        }
        
        # Get weather data if location is available
        if location.get('city'):
            try:
                weather_data = await weather_service.get_current_weather(
                    location['city'], 
                    location.get('state')
                )
                context['current_weather'] = weather_data
            except Exception:
                pass
        
        # Generate response using Ollama
        ollama_available = await ollama_service.check_ollama_status()
        
        if ollama_available:
            # Use Ollama for intelligent response
            ai_response = await ollama_service.generate_response(message, context)
            response_source = 'ollama'
        else:
            # Fallback to existing advisory engine
            fallback_response = advisory_engine.generate_response(
                intent, entities, location, conversation_memory[user_id]
            )
            ai_response = fallback_response.get('response', 'I apologize, but I cannot provide a response right now.')
            response_source = 'fallback'
        
        # Add AI response to memory
        conversation_memory[user_id]['messages'].append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat(),
            'source': response_source
        })
        
        # Translate response if needed
        if language.lower() != 'english':
            try:
                translation_result = await translation_service.translate_text(
                    ai_response, language, 'en'
                )
                ai_response = translation_result.get('translated_text', ai_response)
            except Exception:
                pass
        
        return {
            'response': ai_response,
            'user_id': user_id,
            'language': language,
            'intent': intent,
            'entities': entities,
            'context': context,
            'response_source': response_source,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chatbot service error: {str(e)}")

@app.get("/chat/history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 20):
    """Get chat history for a user"""
    try:
        if user_id not in conversation_memory:
            return {'messages': [], 'total': 0}
        
        messages = conversation_memory[user_id]['messages']
        return {
            'messages': messages[-limit:],
            'total': len(messages),
            'user_id': user_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI Crop Doctor endpoints
@app.post("/disease-detection")
async def detect_plant_disease(request: DiseaseDetectionRequest):
    """AI Crop Doctor - Detect plant diseases from image"""
    try:
        # Decode base64 image
        image_data = decode_base64_to_image(request.image_base64)
        
        # Get weather context if location provided
        weather_context = None
        if request.location and request.location.get('city'):
            try:
                weather_context = await weather_service.get_current_weather(
                    request.location['city'],
                    request.location.get('state')
                )
            except Exception:
                pass
        
        # Use enhanced detection
        detection_result = await detect_disease_enhanced(
            image_data, 
            request.crop_type, 
            weather_context,
            use_advanced_models=True
        )
        
        if 'error' in detection_result:
            raise HTTPException(status_code=400, detail=detection_result['error'])
        
        return detection_result
    
    except Exception as e:
        logger.error(f"Disease detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Disease detection failed: {str(e)}")

@app.post("/disease-detection/enhanced")
async def detect_plant_disease_enhanced(request: DiseaseDetectionRequest, use_ml_models: bool = True):
    """Enhanced AI Crop Doctor with advanced ML models"""
    try:
        # Decode base64 image
        image_data = decode_base64_to_image(request.image_base64)
        
        # Get weather context
        weather_context = None
        if request.location and request.location.get('city'):
            try:
                weather_context = await weather_service.get_current_weather(
                    request.location['city'],
                    request.location.get('state')
                )
            except Exception:
                pass
        
        # Enhanced detection with multiple models
        detection_result = await detect_disease_enhanced(
            image_data,
            request.crop_type,
            weather_context,
            use_advanced_models=use_ml_models
        )
        
        return detection_result
    
    except Exception as e:
        logger.error(f"Enhanced detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/disease-detection/upload")
async def upload_disease_image(
    file: UploadFile = File(...),
    crop_type: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    state: Optional[str] = Form(None)
):
    """Upload image file for disease detection"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Perform detection
        detection_result = detect_disease_from_image(image_data, crop_type)
        
        if 'error' in detection_result:
            raise HTTPException(status_code=400, detail=detection_result['error'])
        
        # Add location context if provided
        if city:
            try:
                weather_data = await weather_service.get_current_weather(city, state)
                detection_result['weather_context'] = weather_data
            except Exception:
                pass
        
        return detection_result
    
    except Exception as e:
        logger.error(f"Image upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.get("/disease-detection/history")
async def get_detection_history(limit: int = 20):
    """Get disease detection history"""
    try:
        history = plant_detector.get_detection_history(limit)
        return {
            'detections': history,
            'total': len(history),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Weather endpoints
@app.post("/weather/current")
async def get_current_weather(request: WeatherRequest):
    """Get current weather data"""
    try:
        weather_data = await weather_service.get_current_weather(
            request.city, request.state, request.country
        )
        return weather_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/weather/forecast")
async def get_weather_forecast(request: WeatherRequest, days: int = 5):
    """Get weather forecast"""
    try:
        forecast_data = await weather_service.get_weather_forecast(
            request.city, request.state, request.country, days
        )
        return forecast_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Translation endpoints
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """Translate text using free translation services"""
    try:
        result = await translation_service.translate_text(
            request.text, request.target_language, request.source_language
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Dataset management endpoints
@app.get("/datasets/available")
async def list_datasets():
    """List available free datasets"""
    try:
        return await dataset_manager.list_available_datasets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/datasets/download/{dataset_name}")
async def download_dataset(dataset_name: str, background_tasks: BackgroundTasks):
    """Download a dataset in the background"""
    try:
        download_path = f"static/datasets/{dataset_name}"
        
        # Start download in background
        background_tasks.add_task(
            dataset_manager.download_dataset,
            dataset_name,
            download_path
        )
        
        return {
            'message': f'Dataset {dataset_name} download started',
            'status': 'initiated',
            'download_path': download_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced crop prediction (existing functionality with improvements)
@app.post("/crop-prediction")
async def enhanced_crop_prediction(request: CropPredictionRequest):
    """Enhanced crop prediction with real-time weather integration"""
    try:
        # Get real-time weather data
        weather_data = await weather_service.get_current_weather(request.city, request.state)
        
        # Use actual weather data for prediction
        model_temp = weather_data['temperature']
        model_humidity = weather_data['humidity']
        
        # Load model
        model_path = 'static/models/crop_recommendation_model.joblib'
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Crop recommendation model not found")
        
        model = load(model_path)
        
        # Prepare input data
        input_data = np.array([[
            request.N, request.P, request.K, 
            model_temp, model_humidity, 
            request.ph, request.rainfall
        ]])
        
        # Make prediction
        prediction_probabilities = model.predict_proba(input_data)
        predicted_class = model.predict(input_data)
        
        # Get top recommendations
        crop_classes = model.classes_.tolist()
        recommendations = []
        
        for i, crop in enumerate(crop_classes):
            probability = float(prediction_probabilities[0][i])
            confidence_score = round(probability * 100, 2)
            recommendations.append({
                'crop': crop,
                'confidence': confidence_score,
                'probability': probability
            })
        
        # Sort by probability and get top 5
        recommendations.sort(key=lambda x: x['probability'], reverse=True)
        top_5 = recommendations[:5]
        
        # Get multilingual support
        if request.language.lower() != 'english':
            try:
                # Translate crop names
                for rec in top_5:
                    translation_result = await translation_service.translate_text(
                        rec['crop'], request.language, 'en'
                    )
                    rec['crop_translated'] = translation_result.get('translated_text', rec['crop'])
            except Exception:
                pass
        
        return {
            'status': 'success',
            'top_crop': top_5[0]['crop'],
            'confidence': top_5[0]['confidence'],
            'recommendations': top_5,
            'weather_data': weather_data,
            'input_parameters': {
                'N': request.N, 'P': request.P, 'K': request.K,
                'temperature': model_temp, 'humidity': model_humidity,
                'ph': request.ph, 'rainfall': request.rainfall,
                'location': f"{request.city}, {request.state}"
            },
            'language': request.language,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Crop prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced yield prediction
@app.post("/yield-prediction") 
async def enhanced_yield_prediction(request: YieldPredictionRequest):
    """Enhanced yield prediction with weather integration"""
    try:
        # Load model and encoders
        model_path = 'static/models/yield_prediction_model.joblib'
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Yield prediction model not found")
        
        # Load encoders
        encoder_paths = {
            'state': 'static/labelencoder/statename_le.joblib',
            'district': 'static/labelencoder/districtname_le.joblib', 
            'season': 'static/labelencoder/season_le.joblib',
            'crop': 'static/labelencoder/crop_le.joblib'
        }
        
        encoders = {}
        for name, path in encoder_paths.items():
            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail=f"{name} encoder not found")
            encoders[name] = load(path)
        
        # Encode inputs with fallback for unknown values
        def safe_encode(value, encoder):
            try:
                return int(encoder.transform([value.lower()])[0])
            except ValueError:
                return abs(hash(value.lower())) % len(encoder.classes_)
        
        encoded_state = safe_encode(request.state, encoders['state'])
        encoded_district = safe_encode(request.city, encoders['district'])
        encoded_season = safe_encode(request.season, encoders['season'])
        encoded_crop = safe_encode(request.crop, encoders['crop'])
        
        # Load model and predict
        model = load(model_path)
        input_data = np.array([[encoded_state, encoded_district, encoded_season, encoded_crop, request.area]])
        
        predicted_yield = model.predict(input_data)[0]
        predicted_production = predicted_yield * request.area
        
        # Get weather context for yield adjustment
        try:
            weather_data = await weather_service.get_current_weather(request.city, request.state)
            # Apply weather-based adjustment (simplified)
            weather_factor = 1.0
            if weather_data['humidity'] > 85:
                weather_factor *= 1.05  # High humidity can be good for some crops
            if weather_data['temperature'] > 35:
                weather_factor *= 0.95  # High temperature stress
            
            adjusted_yield = predicted_yield * weather_factor
            adjusted_production = adjusted_yield * request.area
            
        except Exception:
            weather_data = None
            adjusted_yield = predicted_yield
            adjusted_production = predicted_production
        
        return {
            'predicted_yield': round(float(predicted_yield), 2),
            'predicted_production': round(float(predicted_production), 2),
            'weather_adjusted_yield': round(float(adjusted_yield), 2),
            'weather_adjusted_production': round(float(adjusted_production), 2),
            'weather_data': weather_data,
            'input_parameters': {
                'state': request.state,
                'city': request.city,
                'season': request.season,
                'crop': request.crop,
                'area': request.area
            },
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Yield prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Market Price endpoints
@app.post("/market/prices")
async def get_enhanced_commodity_prices(request: MarketPriceRequest):
    """Get comprehensive market prices with analysis"""
    try:
        location = {
            'state': request.state,
            'city': request.market
        } if request.state or request.market else None
        
        result = await get_enhanced_market_prices(request.commodity, location)
        return result
    
    except Exception as e:
        logger.error(f"Enhanced price data error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/prices/{commodity}")
async def get_commodity_prices_simple(commodity: str, state: str = None, market: str = None):
    """Simple market price endpoint"""
    try:
        location = {
            'state': state,
            'city': market
        } if state or market else None
        
        result = await get_enhanced_market_prices(commodity, location)
        return result
    
    except Exception as e:
        logger.error(f"Price data error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/forecast/{commodity}")
async def get_commodity_price_forecast(commodity: str, days: int = 30):
    """Get price forecast for commodity"""
    try:
        result = await get_price_forecast(commodity, days)
        return result
    
    except Exception as e:
        logger.error(f"Price forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/analytics/{commodity}")
async def get_commodity_analytics(commodity: str, analysis_type: str = "comprehensive"):
    """Get market analytics and insights"""
    try:
        result = await get_market_analytics(commodity, analysis_type)
        return result
    
    except Exception as e:
        logger.error(f"Market analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/market/alerts/set")
async def set_price_alert(request: PriceAlertRequest):
    """Set price alert for commodity"""
    try:
        result = price_alert_system.set_price_alert(
            request.commodity, 
            request.target_price, 
            request.alert_type
        )
        return result
    
    except Exception as e:
        logger.error(f"Price alert error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/alerts/check")
async def check_price_alerts():
    """Check all active price alerts"""
    try:
        result = await price_alert_system.check_alerts()
        return {
            'triggered_alerts': result,
            'count': len(result),
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Alert check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ollama management endpoints
@app.get("/ollama/status")
async def ollama_status():
    """Check Ollama service status"""
    try:
        status = await ollama_service.check_ollama_status()
        return {
            'ollama_available': status,
            'base_url': ollama_service.base_url,
            'model': ollama_service.model,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ollama/chat")
async def direct_ollama_chat(message: str, context: Optional[Dict] = None):
    """Direct chat with Ollama (for testing)"""
    try:
        if not await ollama_service.check_ollama_status():
            raise HTTPException(status_code=503, detail="Ollama service not available")
        
        response = await ollama_service.generate_response(message, context)
        return {
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced agricultural advisory endpoints
@app.post("/advisory/comprehensive")
async def comprehensive_agricultural_advisory(
    message: str,
    location: Optional[Dict[str, str]] = None,
    crop_type: Optional[str] = None,
    language: str = "english"
):
    """Comprehensive agricultural advisory combining all services"""
    try:
        # Get weather data if location provided
        weather_data = None
        if location and location.get('city'):
            weather_data = await weather_service.get_current_weather(
                location['city'], location.get('state')
            )
        
        # Build comprehensive context
        context = {
            'query': message,
            'location': location,
            'crop_type': crop_type,
            'weather': weather_data,
            'language': language,
            'timestamp': datetime.now().isoformat()
        }
        
        # Use Ollama for comprehensive response
        if await ollama_service.check_ollama_status():
            agricultural_prompt = f"""
            As an expert agricultural advisor, please provide comprehensive advice for this farming query:
            
            Query: {message}
            Crop Type: {crop_type or 'Not specified'}
            Location: {location.get('city', 'Not specified') if location else 'Not specified'}
            Current Weather: {weather_data.get('weather_description', 'Not available') if weather_data else 'Not available'}
            Temperature: {weather_data.get('temperature', 'Unknown') if weather_data else 'Unknown'}°C
            Humidity: {weather_data.get('humidity', 'Unknown') if weather_data else 'Unknown'}%
            
            Please provide:
            1. Direct answer to the query
            2. Weather-specific recommendations if relevant
            3. Best practices for the mentioned crop (if any)
            4. Preventive measures
            5. Additional helpful tips
            
            Keep the response practical and actionable for farmers.
            """
            
            ai_response = await ollama_service.generate_response(agricultural_prompt, context)
        else:
            # Fallback response
            ai_response = "I'm currently offline. Please try again later or contact your local agricultural extension office."
        
        # Translate if needed
        if language.lower() != 'english':
            try:
                translation_result = await translation_service.translate_text(ai_response, language, 'en')
                ai_response = translation_result.get('translated_text', ai_response)
            except Exception:
                pass
        
        return {
            'advisory_response': ai_response,
            'context': context,
            'services_used': {
                'ai_advisor': await ollama_service.check_ollama_status(),
                'weather_service': weather_data is not None,
                'translation_service': language.lower() != 'english'
            },
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Comprehensive advisory error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints (maintaining compatibility with existing frontend)
@app.post("/crop_prediction")
async def legacy_crop_prediction(request: CropPredictionRequest):
    """Legacy crop prediction endpoint for frontend compatibility"""
    return await enhanced_crop_prediction(request)

@app.post("/yield")
async def legacy_yield_prediction(request: YieldPredictionRequest):
    """Legacy yield prediction endpoint"""
    return await enhanced_yield_prediction(request)

@app.get("/weather")
async def legacy_weather(city: str, state: str = None):
    """Legacy weather endpoint"""
    weather_data = await weather_service.get_current_weather(city, state)
    return {
        'status': 'success',
        'weather': weather_data
    }

# Ollama setup and management endpoints
@app.get("/setup/ollama/status")
async def check_ollama_setup_status():
    """Check Ollama setup status"""
    try:
        status = get_ollama_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setup/ollama/install")
async def setup_ollama():
    """Setup Ollama for agricultural use"""
    try:
        result = await setup_ollama_for_agriculture()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setup/model/install/{model_name}")
async def install_model(model_name: str):
    """Install a specific Ollama model"""
    try:
        assistant = OllamaSetupAssistant()
        if model_name in ['fast', 'balanced', 'quality', 'technical']:
            result = await assistant.install_recommended_model(model_name)
        else:
            result = await assistant.manager.pull_model(model_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/setup/models/available")
async def get_available_models():
    """Get available models for agriculture"""
    try:
        manager = OllamaManager()
        return {
            'recommended_models': manager.recommended_models,
            'model_preferences': {
                'fast': 'phi3 - Lightweight, good for basic queries',
                'balanced': 'llama3.2 - Best overall performance',
                'quality': 'llama3.1 - Highest quality responses',
                'technical': 'codellama - Best for calculations'
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/setup/models/installed")
async def get_installed_models():
    """Get list of installed Ollama models"""
    try:
        manager = OllamaManager()
        result = await manager.list_installed_models()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced model management
@app.get("/models/cv/available")
async def get_available_cv_models():
    """Get available computer vision models"""
    try:
        result = await list_cv_models()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/cv/train")
async def train_cv_model(background_tasks: BackgroundTasks, dataset_path: Optional[str] = None):
    """Train computer vision model for disease detection"""
    try:
        # Start training in background
        background_tasks.add_task(train_disease_detection_model, dataset_path)
        
        return {
            'message': 'Model training started in background',
            'status': 'initiated',
            'expected_duration': '10-30 minutes depending on dataset size'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# System status and monitoring
@app.get("/system/status")
async def system_status():
    """Complete system status check"""
    try:
        # Check all services
        ollama_status = await ollama_service.check_ollama_status()
        
        # Check weather service
        try:
            test_weather = await weather_service.get_current_weather("Delhi")
            weather_status = True
        except:
            weather_status = False
        
        # Check translation service
        try:
            test_translation = await translation_service.translate_text("hello", "hi", "en")
            translation_status = test_translation.get('translated_text') is not None
        except:
            translation_status = False
        
        # Check models
        model_status = {
            'crop_recommendation': os.path.exists('static/models/crop_recommendation_model.joblib'),
            'yield_prediction': os.path.exists('static/models/yield_prediction_model.joblib'),
            'disease_detection': os.path.exists('static/models/cv_models/sklearn_disease_detector.joblib'),
            'plantvillage_model': os.path.exists('static/models/plantvillage_model.joblib')
        }
        
        # Check Ollama setup
        ollama_setup_status = get_ollama_status()
        
        return {
            'system_status': 'operational',
            'services': {
                'smart_chatbot': ollama_status,
                'ai_crop_doctor': True,
                'enhanced_crop_doctor': True,
                'weather_service': weather_status,
                'translation_service': translation_status,
                'market_price_service': True,
                'price_alerts': True
            },
            'models': model_status,
            'ollama_setup': ollama_setup_status,
            'uptime': datetime.now().isoformat(),
            'version': "3.0.0"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Smart Farmer FastAPI Backend")
    logger.info("🌾 Enhanced with Smart Chatbot and AI Crop Doctor")
    logger.info("🔗 Backend will be available at: http://localhost:8000")
    logger.info("📱 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
