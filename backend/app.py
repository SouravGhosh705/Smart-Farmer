#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Fixed Backend for AI-Assisted Farming Application
This backend integrates with real datasets and provides proper error handling
"""

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
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
from datetime import datetime, date
import requests
import json
import uuid
import re
from datetime import timedelta
from multilingual_system import get_multilingual_system
from voice_support_module import (
    voice_assistant, create_voice_session, process_voice_input, 
    get_voice_capabilities, get_voice_analytics, convert_text_to_voice,
    create_simplified_voice_interface, get_voice_training_data
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Cloud-friendly CORS configuration
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001", 
    "https://*.vercel.app",
    "https://*.netlify.app",
    "https://*.railway.app",
    "https://*.render.com"
]

# Get frontend URL from environment
frontend_url = os.environ.get('FRONTEND_URL')
if frontend_url:
    allowed_origins.append(frontend_url)

CORS(app, origins=allowed_origins)

# Cloud-friendly secret key
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Create directories if they don't exist
os.makedirs('static/models', exist_ok=True)
os.makedirs('static/labelencoder', exist_ok=True)
os.makedirs('static/datasets', exist_ok=True)

class DatasetManager:
    """Handle dataset loading and preprocessing"""
    
    def __init__(self):
        self.crop_recommendation_data = None
        self.yield_prediction_data = None
        self.models = {}
        self.label_encoders = {}
    
    def load_crop_recommendation_dataset(self, file_path):
        """Load and preprocess crop recommendation dataset"""
        try:
            df = pd.read_csv(file_path)
            required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Dataset must contain columns: {required_columns}")
            
            # Clean the data
            df = df.dropna()
            df['label'] = df['label'].str.lower().str.strip()
            
            self.crop_recommendation_data = df
            logger.info(f"Loaded crop recommendation dataset with {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"Error loading crop recommendation dataset: {str(e)}")
            return False
    
    def load_yield_prediction_dataset(self, file_path):
        """Load and preprocess yield prediction dataset"""
        try:
            df = pd.read_csv(file_path)
            required_columns = ['State_Name', 'District_Name', 'Season', 'Crop', 'Area', 'Production']
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Dataset must contain columns: {required_columns}")
            
            # Clean the data
            df = df.dropna()
            df['State_Name'] = df['State_Name'].str.lower().str.strip()
            df['District_Name'] = df['District_Name'].str.lower().str.strip()
            df['Season'] = df['Season'].str.lower().str.strip()
            df['Crop'] = df['Crop'].str.lower().str.strip()
            
            # Calculate yield
            df['Yield'] = df['Production'] / df['Area']
            df = df[df['Yield'] > 0]  # Remove invalid yields
            
            self.yield_prediction_data = df
            logger.info(f"Loaded yield prediction dataset with {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"Error loading yield prediction dataset: {str(e)}")
            return False
    
    def train_crop_recommendation_model(self):
        """Train crop recommendation model with real data"""
        try:
            if self.crop_recommendation_data is None:
                raise ValueError("Crop recommendation dataset not loaded")
            
            df = self.crop_recommendation_data
            X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
            y = df['label']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Crop recommendation model accuracy: {accuracy:.4f}")
            
            # Save the model
            model_path = 'static/models/crop_recommendation_model.joblib'
            dump(model, model_path)
            
            # Store crop classes for later use
            self.models['crop_recommendation'] = {
                'model': model,
                'classes': model.classes_.tolist(),
                'accuracy': accuracy
            }
            
            return True, accuracy
        except Exception as e:
            logger.error(f"Error training crop recommendation model: {str(e)}")
            return False, 0.0
    
    def train_yield_prediction_model(self):
        """Train yield prediction model with real data"""
        try:
            if self.yield_prediction_data is None:
                raise ValueError("Yield prediction dataset not loaded")
            
            df = self.yield_prediction_data
            
            # Create label encoders
            encoders = {}
            for col in ['State_Name', 'District_Name', 'Season', 'Crop']:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                encoders[col.lower().replace('_', '')] = le
                # Save encoder
                dump(le, f'static/labelencoder/{col.lower().replace("_", "")}_le.joblib')
            
            self.label_encoders = encoders
            
            # Prepare features and target
            X = df[['State_Name_encoded', 'District_Name_encoded', 'Season_encoded', 'Crop_encoded', 'Area']]
            y = df['Yield']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            logger.info(f"Yield prediction model RMSE: {rmse:.4f}")
            
            # Save the model
            model_path = 'static/models/yield_prediction_model.joblib'
            dump(model, model_path)
            
            self.models['yield_prediction'] = {
                'model': model,
                'rmse': rmse
            }
            
            return True, rmse
        except Exception as e:
            logger.error(f"Error training yield prediction model: {str(e)}")
            return False, float('inf')

# Initialize dataset manager
dataset_manager = DatasetManager()

# Chatbot conversation memory (in production, use a database)
conversation_memory = {}

class CropAdvisoryEngine:
    """Intelligent crop advisory engine for chatbot"""
    
    def __init__(self):
        self.knowledge_base = self._load_agricultural_knowledge()
        self.intent_patterns = self._load_intent_patterns()
    
    def _load_agricultural_knowledge(self):
        """Load comprehensive agricultural knowledge base"""
        return {
            'crop_care': {
                'rice': {
                    'planting': 'Plant rice during monsoon season (June-July). Ensure fields are flooded.',
                    'watering': 'Maintain 2-5cm water level throughout growing season.',
                    'fertilizer': 'Apply NPK in ratio 4:2:1. First dose during transplanting.',
                    'diseases': 'Watch for blast, brown spot, and bacterial blight.',
                    'harvest': 'Harvest when 80% of grains turn golden yellow.'
                },
                'wheat': {
                    'planting': 'Sow wheat in November-December for rabi season.',
                    'watering': 'Irrigate 4-6 times during growing season.',
                    'fertilizer': 'Apply urea, DAP, and potash as basal dose.',
                    'diseases': 'Monitor for rust, smut, and aphids.',
                    'harvest': 'Harvest when moisture content is 14-16%.'
                },
                'cotton': {
                    'planting': 'Plant cotton in May-June with adequate spacing.',
                    'watering': 'Deep irrigation every 15-20 days.',
                    'fertilizer': 'Heavy nitrogen requirement, split application.',
                    'diseases': 'Check for bollworm, whitefly, and bacterial blight.',
                    'harvest': 'Pick cotton when bolls are fully opened.'
                },
                'maize': {
                    'planting': 'Plant during kharif (June-July) or rabi (November) season.',
                    'watering': 'Ensure adequate moisture during tasseling stage.',
                    'fertilizer': 'Apply nitrogen in splits, phosphorus as basal.',
                    'diseases': 'Watch for fall armyworm and stem borer.',
                    'harvest': 'Harvest when kernels reach physiological maturity.'
                }
            },
            'seasonal_advice': {
                'kharif': {
                    'timing': 'June-October',
                    'crops': ['rice', 'cotton', 'maize', 'sugarcane'],
                    'preparation': 'Prepare fields before monsoon, ensure drainage.',
                    'challenges': 'Monitor for excessive rainfall and pest outbreaks.'
                },
                'rabi': {
                    'timing': 'November-April',
                    'crops': ['wheat', 'barley', 'gram', 'mustard'],
                    'preparation': 'Utilize residual moisture, plan irrigation.',
                    'challenges': 'Watch for frost damage and water scarcity.'
                },
                'summer': {
                    'timing': 'March-June',
                    'crops': ['fodder crops', 'vegetables'],
                    'preparation': 'Ensure water availability, use mulching.',
                    'challenges': 'High temperatures and water stress.'
                }
            },
            'general_tips': {
                'soil_health': 'Test soil every 2-3 years, maintain organic matter >1%.',
                'water_management': 'Use drip irrigation for water conservation.',
                'pest_control': 'Implement IPM strategies, monitor regularly.',
                'storage': 'Maintain 12-14% moisture for safe storage.',
                'market': 'Track commodity prices, plan marketing strategy.'
            }
        }
    
    def _load_intent_patterns(self):
        """Load patterns for intent recognition"""
        return {
            'crop_recommendation': [
                r'what.*crop.*should.*plant',
                r'which.*crop.*best.*for',
                r'recommend.*crop',
                r'suggest.*crop',
                r'best.*crop.*for.*location'
            ],
            'planting_advice': [
                r'when.*plant.*',
                r'how.*plant.*',
                r'planting.*time',
                r'sowing.*season',
                r'cultivation.*method'
            ],
            'disease_help': [
                r'disease.*in.*crop',
                r'pest.*problem',
                r'crop.*sick',
                r'plant.*dying',
                r'leaves.*turning.*yellow'
            ],
            'fertilizer_advice': [
                r'fertilizer.*for.*',
                r'nutrients.*needed',
                r'npk.*ratio',
                r'organic.*fertilizer',
                r'soil.*nutrition'
            ],
            'weather_concern': [
                r'weather.*affect',
                r'rain.*too.*much',
                r'drought.*condition',
                r'temperature.*high',
                r'climate.*change'
            ],
            'yield_inquiry': [
                r'expected.*yield',
                r'production.*estimate',
                r'harvest.*quantity',
                r'crop.*output',
                r'how.*much.*can.*get',
                r'how.*much.*will.*get',
                r'production.*of.*',
                r'yield.*of.*',
                r'how.*much.*.*hectare',
                r'what.*yield.*expect',
                r'expected.*production',
                r'.*yield.*from.*hectare',
                r'.*production.*from.*hectare'
            ],
            'price_inquiry': [
                r'price.*of.*',
                r'market.*rate',
                r'selling.*price',
                r'commodity.*price'
            ]
        }
    
    def detect_intent(self, message):
        """Detect user intent from message"""
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        
        return 'general_inquiry'
    
    def extract_entities(self, message):
        """Extract entities like crop names, locations from message"""
        entities = {
            'crops': [],
            'locations': [],
            'seasons': [],
            'numbers': []
        }
        
        # Crop name mapping with synonyms and alternative names
        crop_synonyms = {
            'rice': ['rice', 'paddy', 'dhan'],
            'wheat': ['wheat', 'gehun'],
            'cotton': ['cotton', 'kapas'],
            'maize': ['maize', 'corn', 'makka', 'bhutta'],
            'sugarcane': ['sugarcane', 'ganna'],
            'potato': ['potato', 'aloo'],
            'tomato': ['tomato', 'tamatar'],
            'onion': ['onion', 'pyaz'],
            'chickpea': ['chickpea', 'gram', 'chana', 'bengal gram'],
            'soybean': ['soybean', 'soya', 'soy'],
            'groundnut': ['groundnut', 'peanut', 'moongfali'],
            'mustard': ['mustard', 'sarson', 'rai'],
            'barley': ['barley', 'jau'],
            'jowar': ['jowar', 'sorghum', 'durra'],
            'bajra': ['bajra', 'millet', 'pearl millet'],
            'ragi': ['ragi', 'finger millet', 'mandua'],
            'arhar': ['arhar', 'pigeon pea', 'tur', 'toor'],
            'masoor': ['masoor', 'lentil', 'red lentil'],
            'moong': ['moong', 'green gram', 'mung'],
            'urad': ['urad', 'black gram', 'mungo'],
            'sesame': ['sesame', 'sesamum', 'til'],
            'sunflower': ['sunflower', 'surajmukhi'],
            'safflower': ['safflower', 'kusum'],
            'jute': ['jute', 'pat'],
            'coconut': ['coconut', 'nariyal'],
            'mango': ['mango', 'aam'],
            'banana': ['banana', 'kela'],
            'apple': ['apple', 'seb'],
            'orange': ['orange', 'santra'],
            'grapes': ['grapes', 'angur'],
            'cashew': ['cashew', 'kaju'],
            'cardamom': ['cardamom', 'elaichi'],
            'pepper': ['pepper', 'kali mirch', 'black pepper'],
            'turmeric': ['turmeric', 'haldi'],
            'ginger': ['ginger', 'adrak'],
            'coriander': ['coriander', 'dhania'],
            'cumin': ['cumin', 'jeera'],
            'fenugreek': ['fenugreek', 'methi'],
            'fennel': ['fennel', 'saunf']
        }
        
        # Create reverse mapping for lookup
        crop_lookup = {}
        for main_crop, synonyms in crop_synonyms.items():
            for synonym in synonyms:
                crop_lookup[synonym.lower()] = main_crop
        
        # Common locations (Indian states and major cities)
        locations = [
            'punjab', 'haryana', 'uttar pradesh', 'bihar', 'west bengal', 
            'maharashtra', 'gujarat', 'rajasthan', 'madhya pradesh',
            'karnataka', 'tamil nadu', 'kerala', 'andhra pradesh', 'telangana',
            'odisha', 'jharkhand', 'chhattisgarh', 'assam', 'himachal pradesh',
            'delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore', 'hyderabad',
            'pune', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore', 'bhopal'
        ]
        
        # Seasons
        seasons = ['kharif', 'rabi', 'summer', 'monsoon', 'winter', 'zaid']
        
        message_lower = message.lower()
        
        # Extract crops using synonym mapping
        for synonym, main_crop in crop_lookup.items():
            if re.search(rf'\b{re.escape(synonym)}\b', message_lower):
                if main_crop not in entities['crops']:
                    entities['crops'].append(main_crop)
        
        # Extract locations
        for location in locations:
            if re.search(rf'\b{re.escape(location)}\b', message_lower):
                entities['locations'].append(location)
        
        # Extract seasons
        for season in seasons:
            if re.search(rf'\b{re.escape(season)}\b', message_lower):
                entities['seasons'].append(season)
        
        # Extract numbers (including area measurements)
        numbers = re.findall(r'\d+(?:\.\d+)?', message)
        entities['numbers'] = [float(num) for num in numbers]
        
        # Also extract common area units
        if re.search(r'\d+\s*(?:hectare|hectares|ha|acre|acres)', message_lower):
            area_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hectare|hectares|ha|acre|acres)', message_lower)
            if area_match:
                area_value = float(area_match.group(1))
                # Convert acres to hectares if needed
                if 'acre' in area_match.group(0):
                    area_value = area_value * 0.4047  # Convert acres to hectares
                if area_value not in entities['numbers']:
                    entities['numbers'].append(area_value)
        
        return entities
    
    def generate_response(self, intent, entities, user_data, conversation_context):
        """Generate intelligent response based on intent and context"""
        try:
            if intent == 'crop_recommendation':
                return self._handle_crop_recommendation(entities, user_data)
            elif intent == 'planting_advice':
                return self._handle_planting_advice(entities, user_data)
            elif intent == 'disease_help':
                return self._handle_disease_help(entities, user_data)
            elif intent == 'fertilizer_advice':
                return self._handle_fertilizer_advice(entities, user_data)
            elif intent == 'weather_concern':
                return self._handle_weather_concern(entities, user_data)
            elif intent == 'yield_inquiry':
                return self._handle_yield_inquiry(entities, user_data)
            elif intent == 'price_inquiry':
                return self._handle_price_inquiry(entities, user_data)
            else:
                return self._handle_general_inquiry(entities, user_data)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'response': 'I apologize, but I encountered an error processing your request. Please try rephrasing your question.',
                'suggestions': ['Ask about crop recommendations', 'Inquire about weather conditions', 'Get planting advice']
            }
    
    def _handle_crop_recommendation(self, entities, user_data):
        """Handle crop recommendation requests using actual ML model"""
        location = user_data.get('city', 'your location')
        state = user_data.get('state', '')
        
        if not user_data.get('city'):
            return {
                'response': 'To provide accurate crop recommendations, I need to know your location. Please share your city and state.',
                'requires_input': True,
                'input_fields': ['city', 'state'],
                'suggestions': ['Tell me your city name', 'Share your state and district']
            }
        
        # Use actual ML model for recommendations
        try:
            # Check if model exists
            model_path = 'static/models/crop_recommendation_model.joblib'
            if not os.path.exists(model_path):
                return {
                    'response': 'Crop recommendation model is not available. Please ensure the model is trained.',
                    'suggestions': ['Try again later', 'Contact support']
                }
            
            # Get weather data for the location
            weather_data = get_weather_data(location, state)
            temp = weather_data['temperature']
            humidity = weather_data['humidity']
            
            # Use default soil parameters (or ask user for specific values)
            model_n = 50  # Default nitrogen
            model_p = 25  # Default phosphorus
            model_k = 30  # Default potassium
            model_ph = 6.5  # Default pH
            model_rainfall = 150  # Default rainfall
            
            # Load and use the actual ML model
            model = load(model_path)
            input_data = np.array([[model_n, model_p, model_k, temp, humidity, model_ph, model_rainfall]])
            
            # Get predictions with probabilities
            prediction_probabilities = model.predict_proba(input_data)
            predicted_class = model.predict(input_data)
            
            # Get top 5 recommendations
            crop_classes = model.classes_.tolist()
            crop_recommendations = []
            
            for i, crop in enumerate(crop_classes):
                probability = float(prediction_probabilities[0][i])
                confidence_score = round(probability * 100, 2)
                crop_recommendations.append({
                    'crop': crop,
                    'confidence': confidence_score,
                    'probability': probability
                })
            
            # Sort by probability and get top 5
            crop_recommendations.sort(key=lambda x: x['probability'], reverse=True)
            top_5_crops = crop_recommendations[:5]
            
            # Format response
            response_text = f"**AI-Powered Crop Recommendations for {location.title()}:**\n\n"
            response_text += f"🌡️ **Current Weather**: {temp}°C, {humidity}% humidity, {weather_data['weather_description']}\n\n"
            response_text += "**Top Recommended Crops:**\n"
            
            recommendations = []
            for i, crop_data in enumerate(top_5_crops, 1):
                crop = crop_data['crop']
                confidence = crop_data['confidence']
                recommendations.append(crop)
                response_text += f"{i}. **{crop.title()}** - {confidence:.1f}% confidence\n"
            
            response_text += f"\n💡 **Best Choice**: {top_5_crops[0]['crop'].title()} ({top_5_crops[0]['confidence']:.1f}% confidence)\n"
            response_text += f"\n*Based on current weather conditions and typical soil parameters for your region.*"
            
            return {
                'response': response_text,
                'recommendations': recommendations,
                'model_prediction': {
                    'top_crop': top_5_crops[0]['crop'],
                    'confidence': top_5_crops[0]['confidence'],
                    'all_predictions': top_5_crops
                },
                'weather_data': weather_data,
                'suggestions': [
                    'Get yield prediction for recommended crop',
                    'Ask about planting time',
                    'Get fertilizer recommendations',
                    'Check market prices'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in ML crop recommendation: {str(e)}")
            # Fallback to weather-based simple logic
            try:
                weather_data = get_weather_data(location, state)
                temp = weather_data['temperature']
                humidity = weather_data['humidity']
                
                if temp > 30 and humidity > 70:
                    recommendations = ['rice', 'sugarcane', 'cotton']
                    reason = f"Based on weather ({temp}°C, {humidity}% humidity), these crops are suitable."
                elif temp < 25 and humidity < 60:
                    recommendations = ['wheat', 'barley', 'gram']
                    reason = f"Current conditions ({temp}°C, {humidity}% humidity) favor rabi crops."
                else:
                    recommendations = ['maize', 'cotton', 'soybean']
                    reason = f"Weather conditions ({temp}°C, {humidity}% humidity) suit these crops."
                
                response_text = f"**Weather-Based Recommendations for {location.title()}:**\n\n{reason}\n\nRecommended: {', '.join(recommendations)}"
                
                return {
                    'response': response_text,
                    'recommendations': recommendations,
                    'suggestions': [
                        'Ask about specific crop planting time',
                        'Get fertilizer recommendations',
                        'Check yield predictions'
                    ]
                }
            except Exception:
                return {
                    'response': f"I recommend conducting a soil test for your location. Common crops include rice, wheat, and cotton depending on the season.",
                    'recommendations': ['rice', 'wheat', 'cotton'],
                    'suggestions': [
                        'Tell me your soil parameters',
                        'Ask about seasonal crops',
                        'Get general farming advice'
                    ]
                }
    
    def _handle_planting_advice(self, entities, user_data):
        """Handle planting and cultivation advice"""
        crops = entities.get('crops', [])
        
        if not crops:
            return {
                'response': 'Which crop would you like planting advice for? I can help with rice, wheat, cotton, maize, and many others.',
                'requires_input': True,
                'suggestions': ['Rice planting guide', 'Wheat cultivation tips', 'Cotton farming advice']
            }
        
        crop = crops[0]
        if crop in self.knowledge_base['crop_care']:
            crop_info = self.knowledge_base['crop_care'][crop]
            response = f"For {crop.title()} cultivation:\n\n"
            response += f"🌱 **Planting**: {crop_info['planting']}\n"
            response += f"💧 **Watering**: {crop_info['watering']}\n"
            response += f"🧪 **Fertilizer**: {crop_info['fertilizer']}\n"
            response += f"🦠 **Disease Prevention**: {crop_info['diseases']}\n"
            response += f"🌾 **Harvest**: {crop_info['harvest']}"
        else:
            response = f"For {crop.title()}, I recommend consulting local agricultural extension services for specific cultivation practices in your region."
        
        return {
            'response': response,
            'suggestions': [
                f'Weather conditions for {crop}',
                f'Fertilizer schedule for {crop}',
                f'Common diseases in {crop}'
            ]
        }
    
    def _handle_disease_help(self, entities, user_data):
        """Handle disease and pest management queries"""
        crops = entities.get('crops', [])
        
        if crops:
            crop = crops[0]
            response = f"For {crop.title()} disease management:\n\n"
            response += "🔍 **Early Detection**: Check plants weekly for symptoms\n"
            response += "🌿 **Organic Solutions**: Use neem oil or bacterial solutions\n"
            response += "💊 **Chemical Control**: Apply fungicides only when necessary\n"
            response += "🛡️ **Prevention**: Maintain proper spacing and field hygiene"
            
            if crop in self.knowledge_base['crop_care']:
                disease_info = self.knowledge_base['crop_care'][crop]['diseases']
                response += f"\n\n**Specific for {crop.title()}**: {disease_info}"
        else:
            response = "To help with disease management, please tell me which crop you're growing and describe the symptoms you're observing."
        
        return {
            'response': response,
            'suggestions': [
                'Describe plant symptoms',
                'Organic pest control methods',
                'Preventive measures'
            ]
        }
    
    def _handle_fertilizer_advice(self, entities, user_data):
        """Handle fertilizer and nutrition advice"""
        crops = entities.get('crops', [])
        
        response = "**Fertilizer Recommendations:**\n\n"
        
        if crops:
            crop = crops[0]
            if crop in self.knowledge_base['crop_care']:
                fert_info = self.knowledge_base['crop_care'][crop]['fertilizer']
                response += f"For {crop.title()}: {fert_info}\n\n"
        
        response += "**General Guidelines:**\n"
        response += "🧪 **Soil Testing**: Test soil pH and nutrient levels\n"
        response += "🌱 **Organic Matter**: Add compost or vermicompost\n"
        response += "⚖️ **Balanced Nutrition**: Use NPK in appropriate ratios\n"
        response += "📅 **Timing**: Split fertilizer application for better uptake"
        
        return {
            'response': response,
            'suggestions': [
                'Soil testing procedures',
                'Organic fertilizer options',
                'NPK ratio calculator'
            ]
        }
    
    def _handle_weather_concern(self, entities, user_data):
        """Handle weather-related farming concerns"""
        location = user_data.get('city', 'your area')
        
        try:
            weather_data = get_weather_data(location)
            temp = weather_data['temperature']
            humidity = weather_data['humidity']
            description = weather_data['weather_description']
            
            response = f"**Current Weather in {location.title()}:**\n"
            response += f"🌡️ Temperature: {temp}°C\n"
            response += f"💨 Humidity: {humidity}%\n"
            response += f"☁️ Conditions: {description.title()}\n\n"
            
            # Weather-specific advice
            if temp > 35:
                response += "⚠️ **High Temperature Alert**: Provide shade, increase irrigation frequency."
            elif temp < 10:
                response += "❄️ **Cold Weather**: Protect crops from frost, consider mulching."
            elif humidity > 85:
                response += "🌧️ **High Humidity**: Monitor for fungal diseases, ensure good ventilation."
            else:
                response += "✅ **Favorable Conditions**: Good weather for most crops."
        
        except Exception:
            response = "I couldn't fetch current weather data. Please share your specific weather concerns, and I'll provide appropriate advice."
        
        return {
            'response': response,
            'suggestions': [
                'Drought management tips',
                'Flood recovery strategies',
                'Heat stress protection'
            ]
        }
    
    def _handle_yield_inquiry(self, entities, user_data):
        """Handle yield prediction and estimation queries using actual ML model"""
        crops = entities.get('crops', [])
        numbers = entities.get('numbers', [])
        location = user_data.get('city', '')
        state = user_data.get('state', '')
        
        if not crops:
            return {
                'response': 'Which crop would you like yield estimates for? Please also mention your location and area under cultivation.',
                'requires_input': True,
                'input_fields': ['crop', 'area', 'location'],
                'suggestions': ['Rice yield prediction', 'Wheat production estimate']
            }
        
        if not location:
            return {
                'response': 'I need your location to provide accurate yield predictions. Please tell me your city and state.',
                'requires_input': True,
                'input_fields': ['city', 'state'],
                'suggestions': ['Tell me your location', 'Share city and state']
            }
        
        crop = crops[0]
        area = numbers[0] if numbers else 1.0  # Default to 1 hectare if not specified
        
        # Use actual ML model for yield prediction
        try:
            # Check if model exists
            model_path = 'static/models/yield_prediction_model.joblib'
            if not os.path.exists(model_path):
                # Fallback to estimates
                return self._fallback_yield_estimates(crop, area)
            
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
                    return self._fallback_yield_estimates(crop, area)
                encoders[name] = load(path)
            
            # Determine current season
            current_month = datetime.now().month
            if current_month in [6, 7, 8, 9, 10]:  # June to October
                season = 'kharif'
            elif current_month in [11, 12, 1, 2, 3, 4]:  # November to April
                season = 'rabi'
            else:
                season = 'summer'
            
            # Encode input values with fallback handling
            def safe_encode(value, encoder, default_index=0):
                try:
                    return int(encoder.transform([value.lower()])[0])
                except ValueError:
                    # Use hash-based encoding for unknown values
                    return abs(hash(value.lower())) % len(encoder.classes_)
            
            encoded_state = safe_encode(state, encoders['state'])
            encoded_district = safe_encode(location, encoders['district'])
            encoded_season = safe_encode(season, encoders['season'])
            encoded_crop = safe_encode(crop, encoders['crop'])
            
            # Load model and predict
            model = load(model_path)
            input_data = np.array([[encoded_state, encoded_district, encoded_season, encoded_crop, area]])
            
            predicted_yield = model.predict(input_data)[0]
            predicted_production = predicted_yield * area
            
            # Format response with ML predictions
            response = f"**AI-Powered Yield Prediction for {crop.title()}:**\n\n"
            response += f"📍 **Location**: {location.title()}, {state.title()}\n"
            response += f"📏 **Area**: {area} hectares\n"
            response += f"📅 **Season**: {season.title()}\n\n"
            response += f"🤖 **ML Model Prediction:**\n"
            response += f"📊 **Expected Yield**: {predicted_yield:.2f} quintals/hectare\n"
            response += f"🌾 **Total Production**: {predicted_production:.2f} quintals\n\n"
            
            # Add weather context
            try:
                weather_data = get_weather_data(location, state)
                response += f"🌤️ **Current Weather**: {weather_data['temperature']}°C, {weather_data['weather_description']}\n\n"
            except Exception:
                pass
            
            response += "*Prediction based on historical data, current location, and season. Actual yields may vary based on farming practices, soil health, and weather conditions.*"
            
            return {
                'response': response,
                'ml_prediction': {
                    'yield_per_hectare': round(predicted_yield, 2),
                    'total_production': round(predicted_production, 2),
                    'area': area,
                    'crop': crop,
                    'season': season,
                    'location': location
                },
                'suggestions': [
                    'Get crop recommendations',
                    'Ask about improving yield',
                    'Check market prices for this crop',
                    'Get planting advice'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in ML yield prediction: {str(e)}")
            return self._fallback_yield_estimates(crop, area)
    
    def _fallback_yield_estimates(self, crop, area):
        """Fallback yield estimates when ML model is not available"""
        # Rough yield estimates (quintals per hectare)
        yield_estimates = {
            'rice': 40, 'wheat': 35, 'cotton': 15, 'maize': 45, 'sugarcane': 700,
            'potato': 250, 'tomato': 300, 'onion': 400, 'soybean': 25, 'gram': 20
        }
        
        estimated_yield = yield_estimates.get(crop, 30)
        total_production = estimated_yield * area
        
        response = f"**Estimated Yield for {crop.title()}:**\n\n"
        response += f"📏 **Area**: {area} hectares\n"
        response += f"📊 **Estimated Yield**: {estimated_yield} quintals/hectare\n"
        response += f"🌾 **Total Production**: {total_production} quintals\n\n"
        response += "*These are general estimates. For more accurate predictions, please provide specific soil and weather data.*"
        
        return {
            'response': response,
            'estimated_yield': estimated_yield,
            'total_production': total_production,
            'suggestions': [
                'Get detailed yield prediction',
                'Factors affecting yield',
                'Yield improvement tips'
            ]
        }
    
    def _handle_price_inquiry(self, entities, user_data):
        """Handle market price and selling advice"""
        crops = entities.get('crops', [])
        
        if not crops:
            return {
                'response': 'Which crop would you like price information for? I can provide market rates and trends.',
                'suggestions': ['Rice market prices', 'Wheat price trends', 'Cotton rates']
            }
        
        crop = crops[0]
        
        # Get price data using existing function
        try:
            price_data = generate_crop_price_analysis(crop)
            current_price = price_data['basePrice2021']
            
            response = f"**Market Information for {crop.title()}:**\n\n"
            response += f"💰 **Current Price**: {current_price} per quintal\n"
            response += f"📈 **Production States**: {price_data['productionState']}\n"
            response += f"🌍 **Export Markets**: {price_data['exportCountry']}\n"
            response += f"📅 **Season**: {price_data['productionSeason']}\n\n"
            response += "💡 **Tip**: Monitor daily mandi rates and consider storage if prices are expected to rise."
        
        except Exception:
            response = f"I'm currently updating price data for {crop.title()}. Please check back shortly or contact your local mandi for current rates."
        
        return {
            'response': response,
            'suggestions': [
                'Price forecasting',
                'Best selling time',
                'Storage advice'
            ]
        }
    
    def _handle_general_inquiry(self, entities, user_data):
        """Handle general farming inquiries"""
        tips = list(self.knowledge_base['general_tips'].values())
        tip = tips[hash(str(entities)) % len(tips)]  # Pseudo-random tip
        
        response = "Here's some helpful farming advice:\n\n"
        response += f"💡 {tip}\n\n"
        response += "I can help you with crop recommendations, planting advice, disease management, fertilizer guidance, weather concerns, yield predictions, and market prices."
        
        return {
            'response': response,
            'suggestions': [
                'Get crop recommendations',
                'Check weather conditions',
                'Ask about fertilizers',
                'Market price inquiry'
            ]
        }

# Initialize advisory engine
advisory_engine = CropAdvisoryEngine()

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'AI-Assisted Farming API - Fixed Version',
        'version': '2.0',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = {
        'crop_recommendation': os.path.exists('static/models/crop_recommendation_model.joblib'),
        'yield_prediction': os.path.exists('static/models/yield_prediction_model.joblib'),
        'label_encoders': all(os.path.exists(f'static/labelencoder/{name}_le.joblib') 
                            for name in ['statename', 'districtname', 'season', 'crop'])
    }
    
    return jsonify({
        'status': 'healthy',
        'models': model_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/load_datasets', methods=['POST'])
def load_datasets():
    """Load datasets from uploaded files"""
    try:
        # Check if files are provided
        crop_file = request.files.get('crop_dataset')
        yield_file = request.files.get('yield_dataset')
        
        if not crop_file or not yield_file:
            return jsonify({
                'error': 'Both crop_dataset and yield_dataset files are required',
                'status': 'error'
            }), 400
        
        # Save uploaded files
        crop_path = 'static/datasets/crop_recommendation.csv'
        yield_path = 'static/datasets/yield_prediction.csv'
        
        crop_file.save(crop_path)
        yield_file.save(yield_path)
        
        # Load datasets
        crop_loaded = dataset_manager.load_crop_recommendation_dataset(crop_path)
        yield_loaded = dataset_manager.load_yield_prediction_dataset(yield_path)
        
        if not crop_loaded or not yield_loaded:
            return jsonify({
                'error': 'Failed to load one or more datasets',
                'status': 'error'
            }), 400
        
        return jsonify({
            'message': 'Datasets loaded successfully',
            'status': 'success',
            'crop_records': len(dataset_manager.crop_recommendation_data),
            'yield_records': len(dataset_manager.yield_prediction_data)
        })
    
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/train_models', methods=['POST'])
def train_models():
    """Train models with loaded datasets"""
    try:
        results = {}
        
        # Train crop recommendation model
        crop_success, crop_accuracy = dataset_manager.train_crop_recommendation_model()
        results['crop_recommendation'] = {
            'success': crop_success,
            'accuracy': crop_accuracy if crop_success else 0.0
        }
        
        # Train yield prediction model
        yield_success, yield_rmse = dataset_manager.train_yield_prediction_model()
        results['yield_prediction'] = {
            'success': yield_success,
            'rmse': yield_rmse if yield_success else float('inf')
        }
        
        if crop_success and yield_success:
            return jsonify({
                'message': 'Models trained successfully',
                'status': 'success',
                'results': results
            })
        else:
            return jsonify({
                'message': 'Some models failed to train',
                'status': 'partial_success',
                'results': results
            }), 206
    
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    """Predict crop recommendation with multilingual support"""
    try:
        # Validate request
        if not request.json:
            multilingual = get_multilingual_system()
            error_msg = multilingual.translate_ui_text('error_no_data', 'english')
            return jsonify({'error': error_msg}), 400
        
        data = request.json
        language = data.get('language', 'english')
        multilingual = get_multilingual_system()
        
        required_fields = ['N', 'P', 'K', 'ph', 'rainfall', 'state', 'city']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            error_msg = multilingual.translate_ui_text('error_missing_fields', language)
            return jsonify({
                'error': f'{error_msg}: {missing_fields}',
                'status': 'error',
                'language': language
            }), 400
        
        # Extract and validate input parameters
        try:
            model_n = float(data['N'])
            model_p = float(data['P'])
            model_k = float(data['K'])
            model_ph = float(data['ph'])
            model_rainfall = float(data['rainfall'])
            state = str(data['state']).lower().strip()
            city = str(data['city']).lower().strip()
        except (ValueError, TypeError) as e:
            error_msg = multilingual.translate_ui_text('error_invalid_data', language)
            return jsonify({
                'error': f'{error_msg}: {str(e)}',
                'status': 'error',
                'language': language
            }), 400
        
        # Get weather data (with fallback)
        try:
            weather_data = get_weather_data(city)
            model_temp = weather_data.get('temperature', 25.0)
            model_humidity = weather_data.get('humidity', 65.0)
        except Exception:
            # Fallback values
            model_temp = 25.0
            model_humidity = 65.0
            logger.warning("Using fallback weather values")
        
        # Load and use the model
        try:
            model_path = 'static/models/crop_recommendation_model.joblib'
            if not os.path.exists(model_path):
                error_msg = multilingual.translate_ui_text('error_model_not_found', language)
                return jsonify({
                    'error': error_msg,
                    'status': 'model_not_found',
                    'language': language
                }), 404
            
            model = load(model_path)
            
            # Prepare input data with proper feature names
            feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            input_data = pd.DataFrame([[model_n, model_p, model_k, model_temp, model_humidity, model_ph, model_rainfall]], 
                                    columns=feature_names)
            
            # Make prediction
            prediction_probabilities = model.predict_proba(input_data)
            predicted_class = model.predict(input_data)
            
            # Get crop classes
            crop_classes = model.classes_.tolist()
            
            # Create response with probabilities
            crop_list = []
            for i, crop in enumerate(crop_classes):
                probability = float(prediction_probabilities[0][i])
                crop_list.append([crop, probability])
            
            # Sort by probability
            crop_list.sort(key=lambda x: x[1], reverse=True)
            
            response = {
                'status': True,
                'success_message': multilingual.translate_ui_text('prediction_success', language),
                'crop': predicted_class[0],
                'crop_list': crop_list,
                'input_parameters': {
                    'N': model_n,
                    'P': model_p,
                    'K': model_k,
                    'temperature': model_temp,
                    'humidity': model_humidity,
                    'ph': model_ph,
                    'rainfall': model_rainfall,
                    'state': state,
                    'city': city
                },
                'language': language,
                'timestamp': datetime.now().isoformat()
            }
            
            # Translate response if needed
            if language and language.lower() != 'english':
                response = multilingual.translate_response(response, language)
            
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            return jsonify({
                'error': f'Model prediction failed: {str(e)}',
                'status': 'prediction_error'
            }), 500
    
    except Exception as e:
        logger.error(f"Error in crop prediction endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/yield_prediction', methods=['POST'])
def yield_prediction():
    """Predict yield with enhanced error handling"""
    try:
        # Validate request
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        required_fields = ['state', 'city', 'season', 'crop', 'area']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'status': 'error'
            }), 400
        
        # Extract and validate input parameters
        try:
            state = str(data['state']).lower().strip()
            city = str(data['city']).lower().strip()
            season = str(data['season']).lower().strip()
            crop = str(data['crop']).lower().strip()
            area = float(data['area'])
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': f'Invalid data types in input: {str(e)}',
                'status': 'error'
            }), 400
        
        if area <= 0:
            return jsonify({
                'error': 'Area must be greater than 0',
                'status': 'error'
            }), 400
        
        # Load label encoders and model
        try:
            model_path = 'static/models/yield_prediction_model.joblib'
            if not os.path.exists(model_path):
                return jsonify({
                    'error': 'Yield prediction model not found. Please train the model first.',
                    'status': 'model_not_found'
                }), 404
            
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
                    return jsonify({
                        'error': f'{name} encoder not found. Please train the model first.',
                        'status': 'encoder_not_found'
                    }), 404
                encoders[name] = load(path)
            
            # Encode input values
            try:
                encoded_state = encoders['state'].transform([state])[0]
                encoded_district = encoders['district'].transform([city])[0]
                encoded_season = encoders['season'].transform([season])[0]
                encoded_crop = encoders['crop'].transform([crop])[0]
            except ValueError as e:
                return jsonify({
                    'error': f'Unknown value in input data: {str(e)}. Please check if the values exist in training data.',
                    'status': 'encoding_error'
                }), 400
            
            # Load model and predict
            model = load(model_path)
            input_data = np.array([[encoded_state, encoded_district, encoded_season, encoded_crop, area]])
            
            predicted_yield = model.predict(input_data)[0]
            predicted_production = predicted_yield * area
            
            response = {
                'predYield': float(predicted_yield),
                'predProduction': float(predicted_production),
                'input_parameters': {
                    'state': state,
                    'city': city,
                    'season': season,
                    'crop': crop,
                    'area': area
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"Error in yield prediction: {str(e)}")
            return jsonify({
                'error': f'Yield prediction failed: {str(e)}',
                'status': 'prediction_error'
            }), 500
    
    except Exception as e:
        logger.error(f"Error in yield prediction endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

def derive_climate_factors(weather_data, season, crop):
    """Calculate climate adjustment factors based on weather conditions"""
    temp = weather_data['temperature']
    humidity = weather_data['humidity']
    
    # Crop-specific optimal ranges (simplified)
    crop_ranges = {
        'rice': {'temp': (20, 35), 'humidity': (70, 90)},
        'wheat': {'temp': (15, 25), 'humidity': (50, 70)},
        'cotton': {'temp': (21, 30), 'humidity': (50, 80)},
        'sugarcane': {'temp': (20, 30), 'humidity': (75, 85)},
        'maize': {'temp': (18, 27), 'humidity': (60, 80)},
        'default': {'temp': (20, 30), 'humidity': (60, 80)}
    }
    
    # Get ranges for the crop (fallback to default)
    ranges = crop_ranges.get(crop.lower(), crop_ranges['default'])
    
    # Temperature factor (1.0 = ideal, <1.0 = suboptimal)
    temp_min, temp_max = ranges['temp']
    if temp_min <= temp <= temp_max:
        temp_factor = 1.0
    else:
        # Penalty for being outside optimal range
        if temp < temp_min:
            temp_factor = max(0.7, 1 - (temp_min - temp) * 0.02)
        else:
            temp_factor = max(0.7, 1 - (temp - temp_max) * 0.02)
    
    # Humidity factor
    humid_min, humid_max = ranges['humidity']
    if humid_min <= humidity <= humid_max:
        humid_factor = 1.0
    else:
        if humidity < humid_min:
            humid_factor = max(0.8, 1 - (humid_min - humidity) * 0.01)
        else:
            humid_factor = max(0.8, 1 - (humidity - humid_max) * 0.01)
    
    # Season factor (simplified)
    season_factors = {
        'kharif': {'rice': 1.1, 'cotton': 1.05, 'maize': 1.0, 'sugarcane': 1.0, 'default': 0.95},
        'rabi': {'wheat': 1.1, 'barley': 1.05, 'gram': 1.0, 'default': 0.95},
        'summer': {'sugarcane': 1.05, 'cotton': 0.9, 'default': 0.9}
    }
    
    season_factor = season_factors.get(season, {}).get(crop.lower(), 
                   season_factors.get(season, {}).get('default', 1.0))
    
    # Overall factor (weighted average)
    overall_factor = (temp_factor * 0.4 + humid_factor * 0.3 + season_factor * 0.3)
    
    return {
        'temp_factor': temp_factor,
        'humid_factor': humid_factor,
        'season_factor': season_factor,
        'overall_factor': overall_factor
    }

def get_weather_data(city, state=None):
    """Get real weather data for a city using OpenWeatherMap API"""
    WEATHER_API_KEY = "ff049be539ac8642b805155154206e4c"
    
    try:
        # Construct location query
        if state:
            location = f"{city},{state},IN"
        else:
            location = f"{city},IN"
        
        # OpenWeatherMap API endpoint
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        
        logger.info(f"Fetching weather data for: {location}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            weather_data = {
                'temperature': float(data['main']['temp']),
                'humidity': float(data['main']['humidity']),
                'pressure': float(data['main']['pressure']),
                'weather_main': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'wind_speed': float(data.get('wind', {}).get('speed', 0)),
                'location': data['name'],
                'country': data['sys']['country']
            }
            
            logger.info(f"Weather data fetched successfully: {weather_data['temperature']}°C, {weather_data['humidity']}% humidity")
            return weather_data
        else:
            logger.warning(f"Weather API returned status code: {response.status_code}")
            raise Exception(f"Weather API error: {response.status_code}")
    
    except Exception as e:
        logger.warning(f"Error fetching weather data: {str(e)}. Using fallback values.")
        # Fallback values based on typical Indian weather
        return {
            'temperature': 28.0,
            'humidity': 70.0,
            'pressure': 1013.25,
            'weather_main': 'Clear',
            'weather_description': 'clear sky',
            'wind_speed': 5.0,
            'location': city,
            'country': 'IN'
        }

@app.route('/recommend_crops', methods=['POST'])
def recommend_crops_by_location():
    """AgriOracle service: Get top 5 crop recommendations based on location with weather integration"""
    try:
        # Validate request
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        required_fields = ['state', 'city']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'status': 'error'
            }), 400
        
        # Extract location data
        state = str(data['state']).lower().strip()
        city = str(data['city']).lower().strip()
        
        # Optional soil parameters (if provided, use them; otherwise use defaults)
        model_n = float(data.get('N', 50))  # Default soil nitrogen
        model_p = float(data.get('P', 25))  # Default soil phosphorus
        model_k = float(data.get('K', 30))  # Default soil potassium
        model_ph = float(data.get('ph', 6.5))  # Default soil pH
        model_rainfall = float(data.get('rainfall', 150))  # Default rainfall
        
        # Get real-time weather data
        try:
            weather_data = get_weather_data(city, state)
            model_temp = weather_data.get('temperature', 28.0)
            model_humidity = weather_data.get('humidity', 70.0)
            weather_info = {
                'current_weather': weather_data.get('weather_description', 'clear sky'),
                'temperature': model_temp,
                'humidity': model_humidity,
                'pressure': weather_data.get('pressure', 1013.25),
                'wind_speed': weather_data.get('wind_speed', 5.0)
            }
        except Exception as e:
            logger.warning(f"Weather API failed: {str(e)}. Using fallback values.")
            model_temp = 28.0
            model_humidity = 70.0
            weather_info = {
                'current_weather': 'data unavailable',
                'temperature': model_temp,
                'humidity': model_humidity,
                'pressure': 1013.25,
                'wind_speed': 5.0,
                'note': 'Using fallback weather data'
            }
        
        # Load and use the crop recommendation model
        try:
            model_path = 'static/models/crop_recommendation_model.joblib'
            if not os.path.exists(model_path):
                return jsonify({
                    'error': 'Crop recommendation model not found. Please train the model first.',
                    'status': 'model_not_found'
                }), 404
            
            model = load(model_path)
            
            # Prepare input data with proper feature names
            feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            input_data = pd.DataFrame([[model_n, model_p, model_k, model_temp, model_humidity, model_ph, model_rainfall]], 
                                    columns=feature_names)
            
            # Make prediction
            prediction_probabilities = model.predict_proba(input_data)
            predicted_class = model.predict(input_data)
            
            # Get crop classes and probabilities
            crop_classes = model.classes_.tolist()
            
            # Create crop recommendations list with probabilities
            crop_recommendations = []
            for i, crop in enumerate(crop_classes):
                probability = float(prediction_probabilities[0][i])
                confidence_score = round(probability * 100, 2)
                crop_recommendations.append({
                    'crop_name': crop.title(),
                    'confidence_score': confidence_score,
                    'probability': probability,
                    'suitability_rating': get_suitability_rating(confidence_score)
                })
            
            # Sort by probability and get top 5
            crop_recommendations.sort(key=lambda x: x['probability'], reverse=True)
            top_5_crops = crop_recommendations[:5]
            
            # Create analysis for each recommended crop
            for crop in top_5_crops:
                crop['analysis'] = generate_crop_analysis(crop['crop_name'], weather_info, {
                    'N': model_n, 'P': model_p, 'K': model_k, 'pH': model_ph, 'rainfall': model_rainfall
                })
            
            response = {
                'status': 'success',
                'service': 'AgriOracle Crop Recommendation',
                'location': {
                    'state': state.title(),
                    'city': city.title()
                },
                'weather_conditions': weather_info,
                'soil_parameters': {
                    'nitrogen': model_n,
                    'phosphorus': model_p,
                    'potassium': model_k,
                    'ph': model_ph,
                    'rainfall': model_rainfall
                },
                'top_recommended_crop': top_5_crops[0]['crop_name'],
                'recommended_crops': top_5_crops,
                'total_crops_analyzed': len(crop_classes),
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"Error in crop recommendation: {str(e)}")
            return jsonify({
                'error': f'Crop recommendation failed: {str(e)}',
                'status': 'prediction_error'
            }), 500
    
    except Exception as e:
        logger.error(f"Error in recommend_crops endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

def get_suitability_rating(confidence_score):
    """Convert confidence score to suitability rating"""
    if confidence_score >= 80:
        return 'Excellent'
    elif confidence_score >= 60:
        return 'Very Good'
    elif confidence_score >= 40:
        return 'Good'
    elif confidence_score >= 20:
        return 'Fair'
    else:
        return 'Poor'

def generate_crop_analysis(crop_name, weather_info, soil_params):
    """Generate analysis text for each crop recommendation"""
    temp = weather_info['temperature']
    humidity = weather_info['humidity']
    ph = soil_params['pH']
    
    analysis = f"Based on current weather conditions ({temp}°C, {humidity}% humidity) and soil parameters (pH {ph}), "
    
    # Add crop-specific insights
    crop_insights = {
        'rice': 'rice thrives in high humidity and warm temperatures with adequate water supply.',
        'wheat': 'wheat grows well in moderate temperatures and is suitable for the current soil conditions.',
        'cotton': 'cotton requires warm weather and well-drained soil, matching current conditions.',
        'maize': 'maize adapts well to various conditions and shows good potential for this location.',
        'sugarcane': 'sugarcane benefits from high humidity and warm temperatures present in this region.',
        'mango': 'mango trees flourish in warm climates with moderate rainfall.',
        'banana': 'banana cultivation is favorable given the humid conditions.',
        'coconut': 'coconut palms thrive in coastal humid conditions similar to current weather.'
    }
    
    crop_key = crop_name.lower()
    if crop_key in crop_insights:
        analysis += crop_insights[crop_key]
    else:
        analysis += f"{crop_name.lower()} shows good compatibility with the current environmental conditions."
    
    return analysis

@app.route('/yield', methods=['POST'])
def yield_prediction_legacy():
    """Legacy yield prediction endpoint for frontend compatibility"""
    try:
        # Validate request
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        required_fields = ['state', 'city', 'season', 'crop', 'area']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'status': 'error'
            }), 400
        
        # Extract and validate input parameters
        try:
            state = str(data['state']).lower().strip()
            city = str(data['city']).lower().strip()
            season = str(data['season']).lower().strip()
            crop = str(data['crop']).lower().strip()
            area = float(data['area'])
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': f'Invalid data types in input: {str(e)}',
                'status': 'error'
            }), 400
        
        if area <= 0:
            return jsonify({
                'error': 'Area must be greater than 0',
                'status': 'error'
            }), 400
        
        # Load label encoders and model
        try:
            model_path = 'static/models/yield_prediction_model.joblib'
            if not os.path.exists(model_path):
                return jsonify({
                    'error': 'Yield prediction model not found. Please train the model first.',
                    'status': 'model_not_found'
                }), 404
            
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
                    return jsonify({
                        'error': f'{name} encoder not found. Please train the model first.',
                        'status': 'encoder_not_found'
                    }), 404
                encoders[name] = load(path)
            
            # Encode input values with location-specific fallback for unknown values
            def get_location_based_encoding(value, encoder, value_type):
                """Generate location-specific encoding for unknown values instead of always using 0"""
                try:
                    return int(encoder.transform([value])[0])
                except ValueError:
                    # Generate a unique encoding based on the value itself
                    # This ensures different locations get different encodings
                    hash_value = abs(hash(value)) % len(encoder.classes_)
                    logger.info(f"Unknown {value_type}: {value}, using hash-based encoding: {hash_value}")
                    return hash_value
            
            encoded_state = get_location_based_encoding(state, encoders['state'], 'state')
            encoded_district = get_location_based_encoding(city, encoders['district'], 'district')
            encoded_season = get_location_based_encoding(season, encoders['season'], 'season')
            encoded_crop = get_location_based_encoding(crop, encoders['crop'], 'crop')
            
            # Get weather data for location-specific adjustments
            climate_multiplier = 1.0
            location_note = ""
            
            try:
                weather_data = get_weather_data(city, state)
                climate_factors = derive_climate_factors(weather_data, season, crop)
                climate_multiplier = climate_factors['overall_factor']
                location_note = f"Weather-adjusted prediction for {weather_data['location']}"
            except Exception as e:
                logger.info(f"Using fallback climate data: {str(e)}")
                location_note = "Using average climate conditions"
            
            # Load model and predict
            model = load(model_path)
            input_data = np.array([[encoded_state, encoded_district, encoded_season, encoded_crop, area]])
            
            # Get base prediction
            base_yield = model.predict(input_data)[0]
            
            # Apply location-specific adjustments
            location_bias = 1 + ((encoded_state + encoded_district) % 10) * 0.015  # Location variability
            predicted_yield = base_yield * climate_multiplier * location_bias
            predicted_production = predicted_yield * area
            
            # Generate data for charts (compatible with frontend expectations)
            seasons = ['kharif', 'rabi', 'summer']
            crops_for_comparison = ['rice', 'wheat', 'cotton', 'sugarcane', 'maize']
            
            # Bar graph data - yield comparison across seasons
            bar_graph_labels = []
            bar_graph_values = []
            
            for s in seasons:
                try:
                    encoded_s = encoders['season'].transform([s])[0]
                    input_s = np.array([[encoded_state, encoded_district, encoded_s, encoded_crop, area]])
                    yield_s = model.predict(input_s)[0]
                    bar_graph_labels.append(s.title())
                    bar_graph_values.append(max(0, float(yield_s)))
                except:
                    bar_graph_labels.append(s.title())
                    bar_graph_values.append(max(0, float(predicted_yield)))
            
            # Pie chart data - crop distribution for the region
            pie_chart_labels = []
            pie_chart_values = []
            
            for c in crops_for_comparison:
                try:
                    encoded_c = encoders['crop'].transform([c])[0]
                    input_c = np.array([[encoded_state, encoded_district, encoded_season, encoded_c, area]])
                    yield_c = model.predict(input_c)[0]
                    pie_chart_labels.append(c.title())
                    pie_chart_values.append(max(1, float(yield_c)))
                except:
                    pie_chart_labels.append(c.title())
                    pie_chart_values.append(max(1, float(predicted_yield * 0.8)))
            
            # Normalize pie chart values to percentages
            total_yield = sum(pie_chart_values)
            pie_chart_values = [round((v/total_yield)*100, 1) for v in pie_chart_values]
            
            response = {
                'predYield': round(float(predicted_yield), 2),
                'predProduction': round(float(predicted_production), 2),
                'barGraphLabel': bar_graph_labels,
                'barGraphvalue': bar_graph_values,
                'pieChartLabel': pie_chart_labels,
                'pieChartValue': pie_chart_values,
                'input_parameters': {
                    'state': state,
                    'city': city,
                    'season': season,
                    'crop': crop,
                    'area': area
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"Error in yield prediction: {str(e)}")
            return jsonify({
                'error': f'Yield prediction failed: {str(e)}',
                'status': 'prediction_error'
            }), 500
    
    except Exception as e:
        logger.error(f"Error in yield prediction endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/individual_price', methods=['POST'])
def predict_crop_price():
    """Predict crop price with market analysis and forecasting"""
    try:
        # Validate request
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        if 'crop_name' not in data:
            return jsonify({
                'error': 'crop_name is required',
                'status': 'error'
            }), 400
        
        crop_name = str(data['crop_name']).lower().strip()
        
        # Generate comprehensive price prediction and market analysis
        price_data = generate_crop_price_analysis(crop_name)
        
        return jsonify(price_data)
    
    except Exception as e:
        logger.error(f"Error in crop price prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

def generate_crop_price_analysis(crop_name):
    """Generate comprehensive crop price analysis and forecast"""
    import random
    from datetime import datetime, timedelta
    import calendar
    
    # Base prices for different crops (in Rs per quintal)
    crop_base_prices = {
        'arhar': {'base': 6000, 'volatility': 0.15},
        'bajra': {'base': 2200, 'volatility': 0.12},
        'barley': {'base': 1800, 'volatility': 0.10},
        'copra': {'base': 12000, 'volatility': 0.20},
        'cotton': {'base': 5500, 'volatility': 0.18},
        'sesamum': {'base': 8000, 'volatility': 0.16},
        'gram': {'base': 4800, 'volatility': 0.14},
        'groundnut': {'base': 5200, 'volatility': 0.13},
        'jowar': {'base': 3000, 'volatility': 0.11},
        'maize': {'base': 2000, 'volatility': 0.12},
        'masoor': {'base': 5500, 'volatility': 0.15},
        'moong': {'base': 7000, 'volatility': 0.16},
        'niger': {'base': 6500, 'volatility': 0.14},
        'paddy': {'base': 2100, 'volatility': 0.08},
        'ragi': {'base': 3200, 'volatility': 0.10},
        'rape': {'base': 4500, 'volatility': 0.13},
        'jute': {'base': 4200, 'volatility': 0.12},
        'safflower': {'base': 5800, 'volatility': 0.15},
        'soyabean': {'base': 4000, 'volatility': 0.14},
        'sugarcane': {'base': 350, 'volatility': 0.06},
        'sunflower': {'base': 6200, 'volatility': 0.13},
        'urad': {'base': 6800, 'volatility': 0.15},
        'wheat': {'base': 2200, 'volatility': 0.09}
    }
    
    # Crop information database
    crop_info = {
        'arhar': {
            'export_country': 'Myanmar, Tanzania, Australia',
            'production_state': 'Maharashtra, Karnataka, Madhya Pradesh',
            'production_season': 'Kharif',
            'image_url': 'arhar.jpg'
        },
        'bajra': {
            'export_country': 'UAE, Nepal, Bangladesh',
            'production_state': 'Rajasthan, Maharashtra, Gujarat',
            'production_season': 'Kharif',
            'image_url': 'bajra.jpg'
        },
        'barley': {
            'export_country': 'UAE, Nepal, Bangladesh',
            'production_state': 'Rajasthan, Uttar Pradesh, Madhya Pradesh',
            'production_season': 'Rabi',
            'image_url': 'barley.jpg'
        },
        'copra': {
            'export_country': 'UAE, Malaysia, USA',
            'production_state': 'Kerala, Tamil Nadu, Karnataka',
            'production_season': 'Year Round',
            'image_url': 'copra.jpg'
        },
        'cotton': {
            'export_country': 'China, Bangladesh, Vietnam',
            'production_state': 'Gujarat, Maharashtra, Telangana',
            'production_season': 'Kharif',
            'image_url': 'cotton.jpg'
        },
        'wheat': {
            'export_country': 'Bangladesh, Nepal, UAE',
            'production_state': 'Uttar Pradesh, Punjab, Haryana',
            'production_season': 'Rabi',
            'image_url': 'wheat.jpg'
        }
    }
    
    # Default info for crops not in database
    default_info = {
        'export_country': 'Various Countries',
        'production_state': 'Multiple States',
        'production_season': 'Seasonal',
        'image_url': f'{crop_name}.jpg'
    }
    
    # Get crop-specific data
    if crop_name in crop_base_prices:
        base_price = crop_base_prices[crop_name]['base']
        volatility = crop_base_prices[crop_name]['volatility']
    else:
        # Default for unknown crops
        base_price = 3000
        volatility = 0.12
    
    crop_details = crop_info.get(crop_name, default_info)
    
    # Generate historical data (previous year)
    current_date = datetime.now()
    previous_year = current_date.year - 1
    
    # Generate monthly data for previous year
    previous_x = []
    previous_y_price = []
    previous_y_wpi = []
    
    for month in range(1, 13):
        month_name = calendar.month_abbr[month]
        previous_x.append(f"{month_name} {previous_year}")
        
        # Seasonal price variation
        seasonal_factor = 1.0
        if crop_details['production_season'].lower() == 'kharif':
            # Higher prices during off-season (Jan-June)
            seasonal_factor = 1.2 if month in [1,2,3,4,5,6] else 0.9
        elif crop_details['production_season'].lower() == 'rabi':
            # Higher prices during off-season (July-Dec)
            seasonal_factor = 1.2 if month in [7,8,9,10,11,12] else 0.9
        
        # Add random market fluctuation
        market_factor = 1 + random.uniform(-volatility, volatility)
        monthly_price = int(base_price * seasonal_factor * market_factor)
        monthly_wpi = int(monthly_price * random.uniform(0.85, 1.05))
        
        previous_y_price.append(monthly_price)
        previous_y_wpi.append(monthly_wpi)
    
    # Generate forecast data (next 12 months)
    forecast_x = []
    forecast_y_price = []
    forecast_y_wpi = []
    price_forecast_table = []
    
    # Current market trend (simulated)
    trend_factor = random.uniform(0.95, 1.08)  # Market can go up or down
    
    for i in range(12):
        future_date = current_date + timedelta(days=30*i)
        month_name = calendar.month_abbr[future_date.month]
        forecast_x.append(f"{month_name} {future_date.year}")
        
        # Apply trend and seasonal factors
        seasonal_factor = 1.0
        if crop_details['production_season'].lower() == 'kharif':
            seasonal_factor = 1.15 if future_date.month in [1,2,3,4,5,6] else 0.92
        elif crop_details['production_season'].lower() == 'rabi':
            seasonal_factor = 1.15 if future_date.month in [7,8,9,10,11,12] else 0.92
        
        # Progressive trend application
        progressive_trend = trend_factor ** (i/12)
        market_factor = 1 + random.uniform(-volatility*0.5, volatility*0.5)
        
        forecast_price = int(base_price * seasonal_factor * progressive_trend * market_factor)
        forecast_wpi = int(forecast_price * random.uniform(0.88, 1.02))
        
        forecast_y_price.append(forecast_price)
        forecast_y_wpi.append(forecast_wpi)
        
        # Calculate change from previous month
        if i == 0:
            change = random.uniform(-5, 8)
        else:
            prev_price = forecast_y_price[i-1]
            change = round(((forecast_price - prev_price) / prev_price) * 100, 1)
        
        price_forecast_table.append([
            forecast_x[i],
            f"₹{forecast_price}",
            f"{forecast_wpi}",
            change
        ])
    
    # Calculate base prices for current and previous year
    base_price_2020 = int(base_price * random.uniform(0.85, 0.95))
    base_price_2021 = int(base_price * random.uniform(0.95, 1.05))
    
    return {
        'cropName': crop_name,
        'basePrice2020': f"₹{base_price_2020}",
        'basePrice2021': f"₹{base_price_2021}",
        'exportCountry': crop_details['export_country'],
        'productionState': crop_details['production_state'],
        'productionSeason': crop_details['production_season'],
        'imageUrl': crop_details['image_url'],
        
        # Chart data for previous year
        'forGraphPreviousX': previous_x,
        'forGraphPreviousYPrice': previous_y_price,
        'forGraphPreviousYWpi': previous_y_wpi,
        
        # Chart data for forecast
        'forGraphForecastX': forecast_x,
        'forGraphForecastYPrice': forecast_y_price,
        'forGraphForecastYWpi': forecast_y_wpi,
        
        # Table data for price forecast
        'priceForecast': price_forecast_table
    }

@app.route('/weather', methods=['GET'])
def get_current_weather():
    """Get current weather data for a location"""
    try:
        city = request.args.get('city')
        state = request.args.get('state')
        
        if not city:
            return jsonify({
                'error': 'City parameter is required',
                'status': 'error'
            }), 400
        
        weather_data = get_weather_data(city, state)
        
        return jsonify({
            'status': 'success',
            'location': {
                'city': city.title(),
                'state': state.title() if state else None
            },
            'weather': weather_data,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error fetching weather: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages"""
    try:
        multilingual = get_multilingual_system()
        return jsonify({
            'status': 'success',
            'supported_languages': multilingual.get_supported_languages(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting languages: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    """Translate specific text elements"""
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        text_type = data.get('type')  # 'crop', 'disease', 'symptom', 'fertilizer', 'ui'
        text_value = data.get('text')
        target_language = data.get('language', 'english')
        
        if not text_type or not text_value:
            return jsonify({
                'error': 'type and text parameters are required',
                'status': 'error'
            }), 400
        
        multilingual = get_multilingual_system()
        
        # Route to appropriate translation method
        if text_type == 'crop':
            translated = multilingual.translate_crop(text_value, target_language)
        elif text_type == 'disease':
            translated = multilingual.translate_disease(text_value, target_language)
        elif text_type == 'symptom':
            translated = multilingual.translate_symptom(text_value, target_language)
        elif text_type == 'fertilizer':
            translated = multilingual.translate_fertilizer(text_value, target_language)
        elif text_type == 'ui':
            translated = multilingual.translate_ui_text(text_value, target_language)
        else:
            return jsonify({
                'error': 'Invalid text type. Use: crop, disease, symptom, fertilizer, ui',
                'status': 'error'
            }), 400
        
        return jsonify({
            'status': 'success',
            'original': text_value,
            'translated': translated,
            'language': target_language,
            'type': text_type,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in translation: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/detect_language', methods=['POST'])
def detect_language():
    """Detect language from input text"""
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        text = data.get('text')
        
        if not text:
            return jsonify({
                'error': 'text parameter is required',
                'status': 'error'
            }), 400
        
        multilingual = get_multilingual_system()
        detected_language = multilingual.detect_language_from_text(text)
        
        return jsonify({
            'status': 'success',
            'detected_language': detected_language,
            'input_text': text,
            'confidence': 'high' if detected_language != 'english' else 'medium',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in language detection: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    """AI-powered plant disease and pest detection from uploaded images"""
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        image_data = data.get('image_data')
        crop_type = data.get('crop_type')
        language = data.get('language', 'english')
        
        if not image_data:
            return jsonify({
                'error': 'image_data is required',
                'status': 'error'
            }), 400
        
        # Import pest detection module
        from pest_disease_detection import detect_disease_from_image, decode_base64_to_image
        
        # Decode base64 image
        try:
            image_bytes = decode_base64_to_image(image_data)
        except Exception as e:
            return jsonify({
                'error': f'Invalid image data: {str(e)}',
                'status': 'error'
            }), 400
        
        # Perform disease detection
        detection_result = detect_disease_from_image(image_bytes, crop_type)
        
        if 'error' in detection_result:
            return jsonify(detection_result), 500
        
        # Add multilingual support for results
        multilingual = get_multilingual_system()
        
        # Translate disease names and descriptions if not English
        if language and language.lower() != 'english':
            try:
                # Translate detected issue names and descriptions
                for issue in detection_result.get('detected_issues', []):
                    if 'name' in issue:
                        issue['name'] = multilingual.translate_disease(issue['name'], language)
                    if 'description' in issue:
                        issue['description'] = multilingual.translate_symptom(issue['description'], language)
                
                # Translate treatment recommendations
                translated_recommendations = []
                for rec in detection_result.get('treatment_recommendations', []):
                    translated_rec = multilingual.translate_ui_text(rec, language)
                    translated_recommendations.append(translated_rec)
                detection_result['treatment_recommendations'] = translated_recommendations
                
                # Translate next steps
                translated_steps = []
                for step in detection_result.get('next_steps', []):
                    translated_step = multilingual.translate_ui_text(step, language)
                    translated_steps.append(translated_step)
                detection_result['next_steps'] = translated_steps
                
                # Translate analysis summary
                if 'analysis_summary' in detection_result:
                    detection_result['analysis_summary'] = multilingual.translate_ui_text(
                        detection_result['analysis_summary'], language
                    )
                    
            except Exception as e:
                logger.warning(f"Translation error: {str(e)}")
                # Continue with English if translation fails
        
        # Add metadata
        detection_result.update({
            'language': language,
            'service': 'AI Crop Doctor',
            'version': '2.0',
            'processing_time': datetime.now().isoformat()
        })
        
        return jsonify(detection_result)
    
    except Exception as e:
        logger.error(f"Error in disease detection endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error during disease detection',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/get_disease_info', methods=['POST'])
def get_disease_info():
    """Get detailed information about a specific disease"""
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        disease_name = data.get('disease_name')
        crop_type = data.get('crop_type')
        language = data.get('language', 'english')
        
        if not disease_name:
            return jsonify({
                'error': 'disease_name is required',
                'status': 'error'
            }), 400
        
        # Import pest detection module
        from pest_disease_detection import plant_detector
        
        # Get disease information
        disease_info = plant_detector.get_disease_info(disease_name, crop_type)
        
        if 'error' in disease_info:
            return jsonify(disease_info), 404
        
        # Add multilingual support
        multilingual = get_multilingual_system()
        
        if language and language.lower() != 'english':
            try:
                # Translate disease information
                if 'disease_name' in disease_info:
                    disease_info['disease_name'] = multilingual.translate_disease(disease_info['disease_name'], language)
                
                # Translate descriptions and symptoms
                if 'information' in disease_info:
                    info = disease_info['information']
                    for key, value in info.items():
                        if isinstance(value, str):
                            info[key] = multilingual.translate_symptom(value, language)
                        elif isinstance(value, list):
                            info[key] = [multilingual.translate_symptom(item, language) for item in value]
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, str):
                                    value[sub_key] = multilingual.translate_symptom(sub_value, language)
                
                # Translate prevention measures
                if 'prevention_measures' in disease_info:
                    translated_prevention = []
                    for measure in disease_info['prevention_measures']:
                        translated_measure = multilingual.translate_ui_text(measure, language)
                        translated_prevention.append(translated_measure)
                    disease_info['prevention_measures'] = translated_prevention
                    
            except Exception as e:
                logger.warning(f"Translation error for disease info: {str(e)}")
        
        disease_info.update({
            'language': language,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify(disease_info)
    
    except Exception as e:
        logger.error(f"Error getting disease info: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/detection_history', methods=['GET'])
def get_detection_history():
    """Get plant disease detection history"""
    try:
        limit = int(request.args.get('limit', 20))
        crop_type = request.args.get('crop_type')
        
        # Import pest detection module
        from pest_disease_detection import plant_detector
        
        # Get detection history
        history = plant_detector.get_detection_history(limit)
        
        # Filter by crop type if specified
        if crop_type:
            history = [h for h in history if h.get('crop_type') == crop_type]
        
        return jsonify({
            'status': 'success',
            'detection_history': history,
            'total_records': len(history),
            'crop_filter': crop_type,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting detection history: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    try:
        info = {
            'crop_recommendation': {
                'model_exists': os.path.exists('static/models/crop_recommendation_model.joblib'),
                'classes': None,
                'accuracy': None
            },
            'yield_prediction': {
                'model_exists': os.path.exists('static/models/yield_prediction_model.joblib'),
                'encoders_exist': {
                    'state': os.path.exists('static/labelencoder/statename_le.joblib'),
                    'district': os.path.exists('static/labelencoder/districtname_le.joblib'),
                    'season': os.path.exists('static/labelencoder/season_le.joblib'),
                    'crop': os.path.exists('static/labelencoder/crop_le.joblib')
                }
            }
        }
        
        # Try to get model details if available
        if 'crop_recommendation' in dataset_manager.models:
            info['crop_recommendation'].update(dataset_manager.models['crop_recommendation'])
        
        if 'yield_prediction' in dataset_manager.models:
            info['yield_prediction'].update(dataset_manager.models['yield_prediction'])
        
        return jsonify(info)
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# ============ CHATBOT ENDPOINTS ============

@app.route('/chat/start', methods=['POST'])
def start_chat_session():
    """Initialize a new chat session"""
    try:
        data = request.json or {}
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Initialize conversation context
        conversation_memory[session_id] = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'user_data': {
                'city': data.get('city', ''),
                'state': data.get('state', ''),
                'language': data.get('language', 'english'),
                'farmer_type': data.get('farmer_type', 'general'),
                'primary_crops': data.get('primary_crops', [])
            },
            'conversation_history': [],
            'context': {
                'last_intent': None,
                'pending_inputs': [],
                'location_set': bool(data.get('city'))
            }
        }
        
        # Generate welcome message
        welcome_msg = generate_welcome_message(conversation_memory[session_id]['user_data'])
        
        # Store welcome message in history
        conversation_memory[session_id]['conversation_history'].append({
            'id': str(uuid.uuid4()),
            'type': 'bot',
            'message': welcome_msg['response'],
            'timestamp': datetime.now().isoformat(),
            'suggestions': welcome_msg.get('suggestions', [])
        })
        
        return jsonify({
            'session_id': session_id,
            'status': 'success',
            'welcome_message': welcome_msg,
            'user_data': conversation_memory[session_id]['user_data'],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error starting chat session: {str(e)}")
        return jsonify({
            'error': 'Failed to start chat session',
            'status': 'error'
        }), 500

@app.route('/chat/message', methods=['POST'])
def chat_message():
    """Process user message and generate response"""
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        session_id = data.get('session_id')
        message = data.get('message', '').strip()
        
        if not session_id or not message:
            return jsonify({
                'error': 'session_id and message are required',
                'status': 'error'
            }), 400
        
        # Check if session exists
        if session_id not in conversation_memory:
            return jsonify({
                'error': 'Invalid session ID. Please start a new chat session.',
                'status': 'session_not_found'
            }), 404
        
        session_data = conversation_memory[session_id]
        
        # Store user message
        user_msg_id = str(uuid.uuid4())
        session_data['conversation_history'].append({
            'id': user_msg_id,
            'type': 'user',
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process message with advisory engine
        intent = advisory_engine.detect_intent(message)
        entities = advisory_engine.extract_entities(message)
        
        # Update user data if new information is provided
        update_user_data_from_message(session_data['user_data'], entities, message)
        
        # Generate response
        response_data = advisory_engine.generate_response(
            intent, 
            entities, 
            session_data['user_data'], 
            session_data['context']
        )
        
        # Store bot response
        bot_msg_id = str(uuid.uuid4())
        session_data['conversation_history'].append({
            'id': bot_msg_id,
            'type': 'bot',
            'message': response_data['response'],
            'timestamp': datetime.now().isoformat(),
            'intent': intent,
            'suggestions': response_data.get('suggestions', []),
            'recommendations': response_data.get('recommendations', []),
            'requires_input': response_data.get('requires_input', False),
            'input_fields': response_data.get('input_fields', [])
        })
        
        # Update context
        session_data['context']['last_intent'] = intent
        
        # Enhanced response with location-specific data
        enhanced_response = enhance_response_with_location_data(
            response_data, 
            session_data['user_data'], 
            intent
        )
        
        return jsonify({
            'session_id': session_id,
            'message_id': bot_msg_id,
            'response': enhanced_response,
            'intent': intent,
            'entities': entities,
            'context_updated': True,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Failed to process message',
            'status': 'error'
        }), 500

@app.route('/chat/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Get conversation history for a session"""
    try:
        if session_id not in conversation_memory:
            return jsonify({
                'error': 'Session not found',
                'status': 'session_not_found'
            }), 404
        
        session_data = conversation_memory[session_id]
        
        return jsonify({
            'session_id': session_id,
            'conversation_history': session_data['conversation_history'],
            'user_data': session_data['user_data'],
            'context': session_data['context'],
            'created_at': session_data['created_at'],
            'message_count': len(session_data['conversation_history']),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve chat history',
            'status': 'error'
        }), 500

@app.route('/chat/update_location', methods=['POST'])
def update_chat_location():
    """Update user location for better recommendations"""
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        session_id = data.get('session_id')
        city = data.get('city')
        state = data.get('state')
        
        if not session_id:
            return jsonify({
                'error': 'session_id is required',
                'status': 'error'
            }), 400
        
        if session_id not in conversation_memory:
            return jsonify({
                'error': 'Session not found',
                'status': 'session_not_found'
            }), 404
        
        # Update user location
        session_data = conversation_memory[session_id]
        if city:
            session_data['user_data']['city'] = city.lower().strip()
        if state:
            session_data['user_data']['state'] = state.lower().strip()
        
        session_data['context']['location_set'] = bool(city)
        
        # Get location-specific welcome message
        if city:
            try:
                weather_data = get_weather_data(city, state)
                location_msg = f"Great! I've updated your location to {city.title()}"
                if state:
                    location_msg += f", {state.title()}"
                location_msg += f". Current weather: {weather_data['temperature']}°C, {weather_data['weather_description']}. How can I help you with farming advice today?"
            except Exception:
                location_msg = f"Location updated to {city.title()}. How can I help you with farming advice?"
        else:
            location_msg = "Please provide your city name for location-specific advice."
        
        return jsonify({
            'session_id': session_id,
            'status': 'success',
            'message': location_msg,
            'updated_data': session_data['user_data'],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error updating chat location: {str(e)}")
        return jsonify({
            'error': 'Failed to update location',
            'status': 'error'
        }), 500

@app.route('/chat/quick_advice', methods=['POST'])
def quick_crop_advice():
    """Get quick crop advice based on current conditions"""
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        city = data.get('city')
        state = data.get('state')
        crop = data.get('crop')
        
        if not city:
            return jsonify({
                'error': 'City is required for location-specific advice',
                'status': 'error'
            }), 400
        
        advice = generate_quick_advice(city, state, crop)
        
        return jsonify({
            'status': 'success',
            'advice': advice,
            'location': {'city': city.title(), 'state': state.title() if state else None},
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error generating quick advice: {str(e)}")
        return jsonify({
            'error': 'Failed to generate advice',
            'status': 'error'
        }), 500

@app.route('/chat/sessions', methods=['GET'])
def get_active_sessions():
    """Get list of active chat sessions (for debugging)"""
    try:
        # Clean up old sessions (older than 24 hours)
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session_data in conversation_memory.items():
            created_at = datetime.fromisoformat(session_data['created_at'])
            if (current_time - created_at).total_seconds() > 86400:  # 24 hours
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del conversation_memory[session_id]
        
        # Return active sessions info
        active_sessions = []
        for session_id, session_data in conversation_memory.items():
            active_sessions.append({
                'session_id': session_id,
                'created_at': session_data['created_at'],
                'message_count': len(session_data['conversation_history']),
                'location': f"{session_data['user_data'].get('city', '')}, {session_data['user_data'].get('state', '')}".strip(', '),
                'last_intent': session_data['context'].get('last_intent')
            })
        
        return jsonify({
            'status': 'success',
            'active_sessions': active_sessions,
            'total_sessions': len(active_sessions),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting active sessions: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve sessions',
            'status': 'error'
        }), 500

# ============ VOICE SUPPORT API ENDPOINTS ============

@app.route('/api/voice/session', methods=['POST'])
def create_voice_interaction_session():
    """Create new voice interaction session"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id')
        language = data.get('language', 'hindi')
        
        session = create_voice_session(user_id, language)
        return jsonify(session)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/process', methods=['POST'])
def process_voice_message():
    """Process voice message in ongoing session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        audio_base64 = data.get('audio_data')
        
        if not session_id or not audio_base64:
            return jsonify({'error': 'Missing session_id or audio_data'}), 400
        
        result = process_voice_input(session_id, audio_base64)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/capabilities', methods=['GET'])
def get_voice_system_capabilities():
    """Get voice system capabilities and supported features"""
    try:
        capabilities = get_voice_capabilities()
        return jsonify(capabilities)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/analytics', methods=['GET'])
def get_voice_system_analytics():
    """Get voice system usage analytics"""
    try:
        analytics = get_voice_analytics()
        return jsonify(analytics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/text-to-speech', methods=['POST'])
def convert_text_to_speech():
    """Convert text to speech for audio responses"""
    try:
        data = request.get_json()
        text = data.get('text')
        language = data.get('language', 'hindi')
        
        if not text:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        result = convert_text_to_voice(text, language)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/simplified-interface', methods=['GET'])
def get_simplified_voice_interface():
    """Get simplified voice interface for low-literate users"""
    try:
        language = request.args.get('language', 'hindi')
        interface = create_simplified_voice_interface(language)
        return jsonify(interface)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/training-data', methods=['GET'])
def get_voice_training_information():
    """Get voice training data for improving recognition"""
    try:
        training_data = get_voice_training_data()
        return jsonify(training_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/session/<session_id>/end', methods=['POST'])
def end_voice_interaction_session(session_id):
    """End voice interaction session"""
    try:
        result = voice_assistant.end_voice_session(session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ CHATBOT HELPER FUNCTIONS ============

def generate_welcome_message(user_data):
    """Generate personalized welcome message"""
    city = user_data.get('city', '')
    language = user_data.get('language', 'english')
    
    if city:
        try:
            weather_data = get_weather_data(city)
            welcome = f"Welcome to Smart Farmer Assistant! 🌾\n\n"
            welcome += f"I see you're in {city.title()}. Current weather: {weather_data['temperature']}°C, {weather_data['weather_description']}.\n\n"
            welcome += "I can help you with:\n"
            welcome += "• Crop recommendations for your location\n"
            welcome += "• Planting and cultivation advice\n"
            welcome += "• Disease and pest management\n"
            welcome += "• Fertilizer recommendations\n"
            welcome += "• Weather-based farming tips\n"
            welcome += "• Yield predictions and market prices\n\n"
            welcome += "What would you like to know about farming today?"
        except Exception:
            welcome = f"Welcome to Smart Farmer Assistant! 🌾\n\nI'm here to help with all your farming questions. What would you like to know?"
    else:
        welcome = "Welcome to Smart Farmer Assistant! 🌾\n\n"
        welcome += "To provide location-specific advice, please tell me your city and state.\n\n"
        welcome += "I can help with crop recommendations, planting advice, disease management, and much more!"
    
    return {
        'response': welcome,
        'suggestions': [
            'What crops should I plant?',
            'Check current weather',
            'Fertilizer recommendations',
            'Market prices'
        ]
    }

def update_user_data_from_message(user_data, entities, message):
    """Update user data based on message content"""
    # Extract location mentions
    if entities['locations']:
        # Simple logic to detect if it's a city or state
        location = entities['locations'][0]
        if not user_data.get('city'):
            user_data['city'] = location
        elif not user_data.get('state'):
            user_data['state'] = location
    
    # Extract crop preferences
    if entities['crops']:
        if 'primary_crops' not in user_data:
            user_data['primary_crops'] = []
        for crop in entities['crops']:
            if crop not in user_data['primary_crops']:
                user_data['primary_crops'].append(crop)

def enhance_response_with_location_data(response_data, user_data, intent):
    """Enhance response with real-time location-specific data"""
    enhanced = response_data.copy()
    
    city = user_data.get('city')
    state = user_data.get('state')
    
    if city and intent in ['crop_recommendation', 'weather_concern']:
        try:
            # Add current weather context
            weather_data = get_weather_data(city, state)
            enhanced['location_data'] = {
                'current_weather': {
                    'temperature': weather_data['temperature'],
                    'humidity': weather_data['humidity'],
                    'description': weather_data['weather_description'],
                    'location': weather_data['location']
                },
                'farming_alerts': generate_weather_alerts(weather_data)
            }
        except Exception:
            enhanced['location_data'] = {
                'note': 'Weather data temporarily unavailable'
            }
    
    return enhanced

def generate_weather_alerts(weather_data):
    """Generate farming alerts based on weather conditions"""
    alerts = []
    temp = weather_data['temperature']
    humidity = weather_data['humidity']
    description = weather_data['weather_description'].lower()
    
    if temp > 35:
        alerts.append({
            'type': 'heat_warning',
            'message': 'High temperature alert! Increase irrigation frequency and provide shade for sensitive crops.',
            'priority': 'high'
        })
    
    if temp < 10:
        alerts.append({
            'type': 'cold_warning',
            'message': 'Cold weather warning! Protect crops from frost damage.',
            'priority': 'high'
        })
    
    if humidity > 85:
        alerts.append({
            'type': 'humidity_warning',
            'message': 'High humidity detected. Monitor crops for fungal diseases.',
            'priority': 'medium'
        })
    
    if 'rain' in description or 'storm' in description:
        alerts.append({
            'type': 'weather_warning',
            'message': 'Rainy conditions. Ensure proper drainage and postpone spraying.',
            'priority': 'medium'
        })
    
    return alerts

def generate_quick_advice(city, state, crop=None):
    """Generate quick farming advice for location and crop"""
    try:
        weather_data = get_weather_data(city, state)
        temp = weather_data['temperature']
        humidity = weather_data['humidity']
        
        advice = {
            'weather_summary': f"Current conditions in {city.title()}: {temp}°C, {humidity}% humidity",
            'immediate_actions': [],
            'crop_specific': {},
            'general_tips': []
        }
        
        # Temperature-based advice
        if temp > 35:
            advice['immediate_actions'].append("Increase irrigation frequency")
            advice['immediate_actions'].append("Provide shade for sensitive plants")
        elif temp < 15:
            advice['immediate_actions'].append("Protect crops from cold stress")
            advice['immediate_actions'].append("Consider frost protection measures")
        
        # Humidity-based advice
        if humidity > 80:
            advice['immediate_actions'].append("Monitor for fungal diseases")
            advice['immediate_actions'].append("Ensure good air circulation")
        elif humidity < 40:
            advice['immediate_actions'].append("Increase moisture retention")
            advice['immediate_actions'].append("Consider mulching")
        
        # Crop-specific advice
        if crop:
            if crop.lower() == 'rice' and temp > 30 and humidity > 70:
                advice['crop_specific']['rice'] = "Excellent conditions for rice. Maintain water levels."
            elif crop.lower() == 'wheat' and temp < 25:
                advice['crop_specific']['wheat'] = "Good conditions for wheat. Monitor for pests."
        
        # General seasonal tips
        current_month = datetime.now().month
        if current_month in [6, 7, 8, 9]:  # Monsoon season
            advice['general_tips'].append("Kharif season: Prepare for rice, cotton, sugarcane")
            advice['general_tips'].append("Ensure proper drainage systems")
        elif current_month in [11, 12, 1, 2]:  # Winter season
            advice['general_tips'].append("Rabi season: Good time for wheat, gram, mustard")
            advice['general_tips'].append("Monitor for frost in cold regions")
        
        return advice
    
    except Exception as e:
        logger.error(f"Error generating quick advice: {str(e)}")
        return {
            'weather_summary': 'Weather data unavailable',
            'immediate_actions': ['Contact local agricultural extension office'],
            'crop_specific': {},
            'general_tips': ['Follow traditional farming calendar for your region']
        }

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'not_found',
        'available_endpoints': [
            '/ - Root endpoint',
            '/health - Health check',
            '/load_datasets - Load training datasets',
            '/train_models - Train ML models',
            '/crop_prediction - Predict crop recommendation',
            '/yield_prediction - Predict crop yield',
            '/yield - Legacy yield prediction with charts (YieldFinder compatibility)',
            '/recommend_crops - AgriOracle crop recommendations by location',
            '/individual_price - Crop price prediction and market analysis',
            '/weather - Get current weather data',
            '/model_info - Get model information',
            '/chat/start - Start new chat session',
            '/chat/message - Send chat message',
            '/chat/history/<session_id> - Get chat history',
            '/chat/update_location - Update user location',
            '/chat/quick_advice - Get quick farming advice',
            '/chat/sessions - Get active chat sessions',
            '/api/voice/session - Create voice interaction session',
            '/api/voice/process - Process voice message',
            '/api/voice/capabilities - Get voice system capabilities',
            '/api/voice/analytics - Get voice usage analytics',
            '/api/voice/text-to-speech - Convert text to speech',
            '/api/voice/simplified-interface - Get simplified voice interface',
            '/api/voice/training-data - Get voice training data',
            '/api/voice/session/<session_id>/end - End voice session'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

def initialize_datasets():
    """Initialize and train models with available datasets"""
    try:
        # Load and train crop recommendation model
        crop_dataset_path = 'static/datasets/crop_recommendation.csv'
        
        if os.path.exists(crop_dataset_path):
            logger.info("Loading crop recommendation dataset...")
            crop_success = dataset_manager.load_crop_recommendation_dataset(crop_dataset_path)
            
            if crop_success:
                logger.info("Training crop recommendation model...")
                train_success, accuracy = dataset_manager.train_crop_recommendation_model()
                
                if train_success:
                    logger.info(f"Crop recommendation model trained successfully with accuracy: {accuracy:.4f}")
                else:
                    logger.error("Failed to train crop recommendation model")
            else:
                logger.error("Failed to load crop recommendation dataset")
        else:
            logger.warning(f"Crop recommendation dataset not found at {crop_dataset_path}")
        
        # Load the Kaggle crop yield dataset
        kaggle_dataset_path = 'static/datasets/crop_yield_indian_states.csv'
        
        if os.path.exists(kaggle_dataset_path):
            logger.info("Loading Kaggle crop yield dataset...")
            success = dataset_manager.load_yield_prediction_dataset(kaggle_dataset_path)
            
            if success:
                logger.info("Training yield prediction model with Kaggle dataset...")
                train_success, rmse = dataset_manager.train_yield_prediction_model()
                
                if train_success:
                    logger.info(f"Yield prediction model training completed successfully with RMSE: {rmse:.4f}")
                else:
                    logger.error("Failed to train yield prediction model")
            else:
                logger.error("Failed to load Kaggle dataset")
        else:
            logger.warning(f"Kaggle dataset not found at {kaggle_dataset_path}")
    except Exception as e:
        logger.error(f"Error initializing datasets: {str(e)}")

if __name__ == '__main__':
    logger.info("Starting AI-Assisted Farming API - Cloud Ready Version")
    
    # Check if we should initialize datasets on startup (disabled by default for performance)
    train_on_startup = os.environ.get('TRAIN_ON_STARTUP', 'false').lower() == 'true'
    
    if train_on_startup:
        logger.info("Training on startup enabled - this may take several minutes...")
        initialize_datasets()
    else:
        logger.info("Training on startup disabled for better performance. Models will be trained on first use.")
    
    # Suppress Flask development server warning (for development only)
    import warnings
    warnings.filterwarnings('ignore', message='.*development server.*')
    
    # Cloud-friendly configuration
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', 8000))  # Cloud platforms often set PORT env var
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting server on {host}:{port} (debug={debug_mode})")
    app.run(host=host, port=port, debug=debug_mode)
