#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Voice Support Module for Smart Farmer Application
Speech-to-text and text-to-speech functionality for low-literate users
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import os
import base64
import io
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VoiceInteraction:
    """Data class for voice interactions"""
    interaction_id: str
    user_speech: str
    recognized_text: str
    intent: str
    response_text: str
    audio_response: Optional[bytes]
    language: str
    confidence_score: float
    timestamp: datetime

class VoiceSupportSystem:
    """Comprehensive voice support system for farmers"""
    
    def __init__(self):
        self.supported_languages = self._load_supported_languages()
        self.voice_commands = self._load_voice_commands()
        self.speech_patterns = self._load_speech_patterns()
        self.audio_responses = self._load_audio_responses()
        os.makedirs('static/voice_cache', exist_ok=True)
        os.makedirs('static/audio_responses', exist_ok=True)
    
    def _load_supported_languages(self) -> Dict:
        """Load supported languages for voice interaction"""
        return {
            'hindi': {
                'language_code': 'hi-IN',
                'name': 'हिंदी',
                'tts_voice': 'hi-IN-Wavenet-A',
                'common_phrases': {
                    'greeting': 'नमस्ते, मैं आपकी खेती में कैसे मदद कर सकता हूं?',
                    'crop_question': 'आप किस फसल के बारे में जानना चाहते हैं?',
                    'location_question': 'आप किस जगह से हैं?',
                    'thank_you': 'धन्यवाद, क्या कोई और सवाल है?'
                }
            },
            'english': {
                'language_code': 'en-IN',
                'name': 'English',
                'tts_voice': 'en-IN-Wavenet-A',
                'common_phrases': {
                    'greeting': 'Hello, how can I help you with farming today?',
                    'crop_question': 'Which crop would you like to know about?',
                    'location_question': 'Where are you located?',
                    'thank_you': 'Thank you, do you have any other questions?'
                }
            },
            'punjabi': {
                'language_code': 'pa-IN',
                'name': 'ਪੰਜਾਬੀ',
                'tts_voice': 'pa-IN-Wavenet-A',
                'common_phrases': {
                    'greeting': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਮੈਂ ਤੁਹਾਡੀ ਖੇਤੀ ਵਿੱਚ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?',
                    'crop_question': 'ਤੁਸੀਂ ਕਿਸ ਫਸਲ ਬਾਰੇ ਜਾਣਨਾ ਚਾਹੁੰਦੇ ਹੋ?',
                    'location_question': 'ਤੁਸੀਂ ਕਿੱਥੋਂ ਹੋ?',
                    'thank_you': 'ਧੰਨਵਾਦ, ਕੀ ਕੋਈ ਹੋਰ ਸਵਾਲ ਹੈ?'
                }
            },
            'bengali': {
                'language_code': 'bn-IN',
                'name': 'বাংলা',
                'tts_voice': 'bn-IN-Wavenet-A',
                'common_phrases': {
                    'greeting': 'নমস্কার, আমি আপনার কৃষিকাজে কীভাবে সাহায্য করতে পারি?',
                    'crop_question': 'আপনি কোন ফসল সম্পর্কে জানতে চান?',
                    'location_question': 'আপনি কোথায় থেকে?',
                    'thank_you': 'ধন্যবাদ, আর কোনো প্রশ্ন আছে?'
                }
            },
            'marathi': {
                'language_code': 'mr-IN',
                'name': 'मराठी',
                'tts_voice': 'mr-IN-Wavenet-A',
                'common_phrases': {
                    'greeting': 'नमस्कार, मी आपल्या शेतीमध्ये कशी मदत करू शकतो?',
                    'crop_question': 'तुम्हाला कोणत्या पिकाविषयी माहिती हवी आहे?',
                    'location_question': 'तुम्ही कुठून आहात?',
                    'thank_you': 'धन्यवाद, काही आणखी प्रश्न आहेत का?'
                }
            },
            'gujarati': {
                'language_code': 'gu-IN',
                'name': 'ગુજરાતી',
                'tts_voice': 'gu-IN-Wavenet-A',
                'common_phrases': {
                    'greeting': 'નમસ્તે, હું તમારી ખેતીમાં કેવી રીતે મદદ કરી શકું?',
                    'crop_question': 'તમે કયા પાકની વિગતો જાણવા માંગો છો?',
                    'location_question': 'તમે ક્યાંથી છો?',
                    'thank_you': 'આભાર, કોઈ અન્ય પ્રશ્નો છે?'
                }
            },
            'tamil': {
                'language_code': 'ta-IN',
                'name': 'தமிழ்',
                'tts_voice': 'ta-IN-Wavenet-A',
                'common_phrases': {
                    'greeting': 'வணக்கம், உங்கள் விவசாயத்தில் நான் எப்படி உதவ முடியும்?',
                    'crop_question': 'எந்த பயிரைப் பற்றி தெரிந்து கொள்ள விரும்புகிறீர்கள்?',
                    'location_question': 'நீங்கள் எங்கிருந்து வருகிறீர்கள்?',
                    'thank_you': 'நன்றி, வேறு கேள்விகள் ஏதும் உள்ளதா?'
                }
            },
            'telugu': {
                'language_code': 'te-IN',
                'name': 'తెలుగు',
                'tts_voice': 'te-IN-Wavenet-A',
                'common_phrases': {
                    'greeting': 'నమస్కారం, మీ వ్యవసాయంలో నేను ఎలా సహాయం చేయగలను?',
                    'crop_question': 'మీరు ఏ పంట గురించి తెలుసుకోవాలని అనుకుంటున్నారు?',
                    'location_question': 'మీరు ఎక్కడ నుండి?',
                    'thank_you': 'ధన్యవాదాలు, ఇంకా ఏమైనా ప్రశ్నలు ఉన్నాయా?'
                }
            }
        }
    
    def _load_voice_commands(self) -> Dict:
        """Load voice commands for navigation and actions"""
        return {
            'navigation': {
                'hindi': {
                    'home': ['घर', 'होम', 'मुख्य पृष्ठ'],
                    'crop_recommendation': ['फसल सुझाव', 'फसल की सिफारिश', 'कौन सी फसल'],
                    'weather': ['मौसम', 'हवा', 'बारिश'],
                    'prices': ['दाम', 'कीमत', 'भाव', 'मंडी'],
                    'help': ['मदद', 'सहायता', 'सहारा']
                },
                'english': {
                    'home': ['home', 'main page', 'dashboard'],
                    'crop_recommendation': ['crop recommendation', 'suggest crop', 'which crop'],
                    'weather': ['weather', 'climate', 'rain', 'temperature'],
                    'prices': ['price', 'rate', 'market', 'mandi'],
                    'help': ['help', 'support', 'assistance']
                },
                'punjabi': {
                    'home': ['ਘਰ', 'ਹੋਮ', 'ਮੁੱਖ ਪੰਨਾ'],
                    'crop_recommendation': ['ਫਸਲ ਸਲਾਹ', 'ਕਿਹੜੀ ਫਸਲ', 'ਫਸਲ ਸੁਝਾਅ'],
                    'weather': ['ਮੌਸਮ', 'ਹਵਾ', 'ਬਾਰਸ਼'],
                    'prices': ['ਭਾਅ', 'ਕੀਮਤ', 'ਮੰਡੀ'],
                    'help': ['ਮਦਦ', 'ਸਹਾਇਤਾ']
                }
            },
            'farming_actions': {
                'hindi': {
                    'sowing': ['बुआई', 'बीज बोना', 'खेत तैयार'],
                    'irrigation': ['सिंचाई', 'पानी', 'सिंचन'],
                    'fertilizer': ['खाद', 'उर्वरक', 'खुराक'],
                    'pesticide': ['दवा', 'कीटनाशक', 'छिड़काव'],
                    'harvesting': ['कटाई', 'फसल काटना', 'बुआई']
                },
                'english': {
                    'sowing': ['sowing', 'planting', 'seeding'],
                    'irrigation': ['irrigation', 'watering', 'water'],
                    'fertilizer': ['fertilizer', 'manure', 'nutrients'],
                    'pesticide': ['pesticide', 'spray', 'medicine'],
                    'harvesting': ['harvesting', 'cutting', 'reaping']
                }
            }
        }
    
    def _load_speech_patterns(self) -> Dict:
        """Load speech recognition patterns for agricultural terms"""
        return {
            'crop_names': {
                'hindi': {
                    'धान': 'rice', 'चावल': 'rice', 'गेहूं': 'wheat', 'गेहुँ': 'wheat',
                    'कपास': 'cotton', 'मक्का': 'maize', 'भुट्टा': 'maize',
                    'गन्ना': 'sugarcane', 'आलू': 'potato', 'टमाटर': 'tomato',
                    'प्याज': 'onion', 'चना': 'gram', 'अरहर': 'pigeon_pea',
                    'सोयाबीन': 'soybean', 'मूंगफली': 'groundnut', 'सरसों': 'mustard'
                },
                'punjabi': {
                    'ਧਾਨ': 'rice', 'ਕਣਕ': 'wheat', 'ਕਪਾਹ': 'cotton',
                    'ਮੱਕੀ': 'maize', 'ਗੰਨਾ': 'sugarcane', 'ਆਲੂ': 'potato',
                    'ਟਮਾਟਰ': 'tomato', 'ਪਿਆਜ਼': 'onion', 'ਚਣਾ': 'gram'
                },
                'english': {
                    'rice': 'rice', 'wheat': 'wheat', 'cotton': 'cotton',
                    'maize': 'maize', 'corn': 'maize', 'sugarcane': 'sugarcane',
                    'potato': 'potato', 'tomato': 'tomato', 'onion': 'onion'
                }
            },
            'weather_terms': {
                'hindi': {
                    'बारिश': 'rain', 'धूप': 'sunny', 'बादल': 'cloudy',
                    'गर्मी': 'hot', 'सर्दी': 'cold', 'ओला': 'hail',
                    'तूफान': 'storm', 'हवा': 'wind'
                },
                'english': {
                    'rain': 'rain', 'sun': 'sunny', 'cloud': 'cloudy',
                    'hot': 'hot', 'cold': 'cold', 'storm': 'storm', 'wind': 'wind'
                }
            },
            'farming_activities': {
                'hindi': {
                    'बुआई': 'sowing', 'कटाई': 'harvesting', 'सिंचाई': 'irrigation',
                    'छिड़काव': 'spraying', 'खुदाई': 'plowing', 'निराई': 'weeding'
                },
                'english': {
                    'sowing': 'sowing', 'harvesting': 'harvesting', 'irrigation': 'irrigation',
                    'spraying': 'spraying', 'plowing': 'plowing', 'weeding': 'weeding'
                }
            }
        }
    
    def _load_audio_responses(self) -> Dict:
        """Load pre-recorded audio responses for common queries"""
        return {
            'hindi': {
                'greeting': 'audio/hindi_greeting.mp3',
                'crop_recommendation': 'audio/hindi_crop_rec.mp3',
                'weather_info': 'audio/hindi_weather.mp3',
                'price_info': 'audio/hindi_prices.mp3'
            },
            'english': {
                'greeting': 'audio/english_greeting.mp3',
                'crop_recommendation': 'audio/english_crop_rec.mp3',
                'weather_info': 'audio/english_weather.mp3',
                'price_info': 'audio/english_prices.mp3'
            }
        }
    
    def process_voice_input(self, audio_data: bytes, language: str = 'hindi') -> VoiceInteraction:
        """Process voice input and generate response"""
        try:
            interaction_id = str(uuid.uuid4())
            
            # Speech-to-text conversion (placeholder for actual implementation)
            recognized_text = self._speech_to_text(audio_data, language)
            
            # Process the recognized text
            intent, entities = self._analyze_voice_input(recognized_text, language)
            
            # Generate text response
            response_text = self._generate_voice_response(intent, entities, language)
            
            # Convert response to speech
            audio_response = self._text_to_speech(response_text, language)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(recognized_text, intent)
            
            interaction = VoiceInteraction(
                interaction_id=interaction_id,
                user_speech="[Audio data received]",
                recognized_text=recognized_text,
                intent=intent,
                response_text=response_text,
                audio_response=audio_response,
                language=language,
                confidence_score=confidence_score,
                timestamp=datetime.now()
            )
            
            # Save interaction for learning
            self._save_voice_interaction(interaction)
            
            return interaction
            
        except Exception as e:
            logger.error(f"Error processing voice input: {str(e)}")
            raise
    
    def _speech_to_text(self, audio_data: bytes, language: str) -> str:
        """Convert speech to text (placeholder for actual STT integration)"""
        try:
            # In real implementation, this would use:
            # - Google Speech-to-Text API
            # - Microsoft Azure Speech Service
            # - AWS Transcribe
            # - Web Speech API (client-side)
            
            # For demonstration, returning simulated recognition results
            sample_texts = {
                'hindi': [
                    'मुझे धान की खेती के बारे में बताइए',
                    'आज का मौसम कैसा है',
                    'गेहूं का भाव क्या है',
                    'कौन सी फसल बोनी चाहिए',
                    'खाद कब डालना चाहिए'
                ],
                'english': [
                    'Tell me about rice cultivation',
                    'What is today\'s weather',
                    'What is wheat price',
                    'Which crop should I grow',
                    'When should I apply fertilizer'
                ],
                'punjabi': [
                    'ਮੈਨੂੰ ਧਾਨ ਦੀ ਖੇਤੀ ਬਾਰੇ ਦੱਸੋ',
                    'ਅੱਜ ਦਾ ਮੌਸਮ ਕਿਹੋ ਜਿਹਾ ਹੈ',
                    'ਕਣਕ ਦਾ ਭਾਅ ਕੀ ਹੈ'
                ]
            }
            
            import random
            return random.choice(sample_texts.get(language, sample_texts['english']))
            
        except Exception as e:
            logger.error(f"Error in speech-to-text: {str(e)}")
            return "Sorry, I couldn't understand that clearly."
    
    def _analyze_voice_input(self, text: str, language: str) -> Tuple[str, Dict]:
        """Analyze voice input to extract intent and entities"""
        try:
            text_lower = text.lower()
            
            # Intent detection based on keywords
            intent = 'general_inquiry'
            entities = {'crops': [], 'locations': [], 'numbers': []}
            
            # Crop detection
            crop_patterns = self.speech_patterns.get('crop_names', {}).get(language, {})
            for local_name, english_name in crop_patterns.items():
                if local_name in text_lower:
                    entities['crops'].append(english_name)
            
            # Intent detection based on content
            if any(word in text_lower for word in ['मौसम', 'weather', 'ਮੌਸਮ', 'बारिश', 'rain']):
                intent = 'weather_inquiry'
            elif any(word in text_lower for word in ['दाम', 'कीमत', 'price', 'भाव', 'ਭਾਅ', 'मंडी']):
                intent = 'price_inquiry'
            elif any(word in text_lower for word in ['फसल', 'crop', 'बोना', 'sowing', 'ਫਸਲ']):
                intent = 'crop_recommendation'
            elif any(word in text_lower for word in ['खाद', 'fertilizer', 'उर्वरक', 'nutrients']):
                intent = 'fertilizer_advice'
            elif any(word in text_lower for word in ['बीमारी', 'disease', 'कीट', 'pest', 'दवा']):
                intent = 'disease_pest_help'
            elif any(word in text_lower for word in ['उत्पादन', 'yield', 'production', 'फसल मिलेगी']):
                intent = 'yield_prediction'
            
            # Extract numbers (area, quantity, etc.)
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            entities['numbers'] = [float(num) for num in numbers]
            
            # Extract locations (simplified)
            common_places = ['delhi', 'mumbai', 'pune', 'lucknow', 'kanpur', 'amritsar']
            for place in common_places:
                if place in text_lower:
                    entities['locations'].append(place)
            
            return intent, entities
            
        except Exception as e:
            logger.error(f"Error analyzing voice input: {str(e)}")
            return 'general_inquiry', {}
    
    def _generate_voice_response(self, intent: str, entities: Dict, language: str) -> str:
        """Generate appropriate voice response"""
        try:
            # Get language-specific phrases
            lang_phrases = self.supported_languages.get(language, {}).get('common_phrases', {})
            
            if intent == 'crop_recommendation':
                if entities.get('locations'):
                    location = entities['locations'][0]
                    if language == 'hindi':
                        response = f"{location} के लिए मैं धान, गेहूं और कपास की सिफारिश करता हूं। आपकी मिट्टी और मौसम के अनुसार धान सबसे अच्छा विकल्प है।"
                    else:
                        response = f"For {location}, I recommend rice, wheat, and cotton. Based on your soil and weather, rice is the best option."
                else:
                    if language == 'hindi':
                        response = "फसल की सिफारिश के लिए मुझे आपका स्थान बताइए। मैं मिट्टी और मौसम के अनुसार सबसे अच्छी फसल सुझाऊंगा।"
                    else:
                        response = "To recommend crops, please tell me your location. I'll suggest the best crops based on soil and weather."
            
            elif intent == 'weather_inquiry':
                if language == 'hindi':
                    response = "आज का मौसम साफ है, तापमान 28 डिग्री और नमी 65 प्रतिशत है। फसल के लिए अच्छा मौसम है।"
                else:
                    response = "Today's weather is clear with temperature 28°C and humidity 65%. Good conditions for farming."
            
            elif intent == 'price_inquiry':
                crops = entities.get('crops', ['wheat'])
                crop = crops[0] if crops else 'wheat'
                if language == 'hindi':
                    response = f"{crop} का आज का भाव 2400 रुपए प्रति क्विंटल है। कल से 50 रुपए बढ़ा है। बेचने का अच्छा समय है।"
                else:
                    response = f"Today's {crop} price is ₹2400 per quintal. It has increased by ₹50 from yesterday. Good time to sell."
            
            elif intent == 'fertilizer_advice':
                crops = entities.get('crops', ['general'])
                if language == 'hindi':
                    response = "खाद डालने के लिए पहले मिट्टी की जांच करवाएं। सामान्यतः यूरिया, डीएपी और पोटाश का उपयोग करें।"
                else:
                    response = "For fertilizer application, first test your soil. Generally use urea, DAP, and potash in recommended doses."
            
            elif intent == 'disease_pest_help':
                if language == 'hindi':
                    response = "पौधे की बीमारी के लिए नीम का तेल छिड़कें। तस्वीर भेजकर सटीक इलाज जान सकते हैं।"
                else:
                    response = "For plant diseases, spray neem oil. Send a photo for accurate diagnosis and treatment."
            
            elif intent == 'yield_prediction':
                crops = entities.get('crops', ['rice'])
                area = entities.get('numbers', [1])[0] if entities.get('numbers') else 1
                crop = crops[0] if crops else 'rice'
                if language == 'hindi':
                    response = f"{area} एकड़ में {crop} से लगभग 40 क्विंटल उत्पादन होगा। मौसम और खाद के अनुसार यह बढ़-घट सकता है।"
                else:
                    response = f"In {area} acres of {crop}, you can expect about 40 quintals production. This may vary based on weather and fertilizer."
            
            else:  # general_inquiry
                response = lang_phrases.get('greeting', 'Hello, how can I help you with farming today?')
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating voice response: {str(e)}")
            return self.supported_languages.get(language, {}).get('common_phrases', {}).get('greeting', 'How can I help you?')
    
    def _text_to_speech(self, text: str, language: str) -> bytes:
        """Convert text to speech (placeholder for actual TTS integration)"""
        try:
            # In real implementation, this would use:
            # - Google Text-to-Speech API
            # - Microsoft Azure Speech Service
            # - AWS Polly
            # - Web Speech API Synthesis (client-side)
            
            # For demonstration, returning empty bytes
            # In real implementation, this would return actual audio data
            logger.info(f"TTS: Converting text to speech in {language}: {text[:50]}...")
            
            # Simulate audio generation
            import time
            time.sleep(0.1)  # Simulate processing time
            
            # Return empty bytes as placeholder
            return b""
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
            return b""
    
    def _calculate_confidence(self, recognized_text: str, intent: str) -> float:
        """Calculate confidence score for voice recognition"""
        try:
            # Simplified confidence calculation
            text_length = len(recognized_text.split())
            
            # Base confidence
            confidence = 0.8
            
            # Adjust based on text clarity indicators
            if text_length < 3:
                confidence -= 0.2
            elif text_length > 20:
                confidence -= 0.1
            
            # Adjust based on intent detection
            if intent == 'general_inquiry':
                confidence -= 0.1  # Lower confidence for generic intent
            
            # Check for agricultural terminology
            agri_terms = ['crop', 'weather', 'price', 'farm', 'फसल', 'मौसम', 'दाम', 'खेत']
            if any(term in recognized_text.lower() for term in agri_terms):
                confidence += 0.1
            
            return max(0.3, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _save_voice_interaction(self, interaction: VoiceInteraction) -> None:
        """Save voice interaction for learning and improvement"""
        try:
            interaction_data = {
                'interaction_id': interaction.interaction_id,
                'recognized_text': interaction.recognized_text,
                'intent': interaction.intent,
                'response_text': interaction.response_text,
                'language': interaction.language,
                'confidence_score': interaction.confidence_score,
                'timestamp': interaction.timestamp.isoformat()
            }
            
            # Save to JSON file (in production, use database)
            interactions_file = 'static/voice_interactions.json'
            
            # Load existing interactions
            if os.path.exists(interactions_file):
                with open(interactions_file, 'r', encoding='utf-8') as f:
                    existing_interactions = json.load(f)
            else:
                existing_interactions = []
            
            # Add new interaction
            existing_interactions.append(interaction_data)
            
            # Keep only last 1000 interactions
            if len(existing_interactions) > 1000:
                existing_interactions = existing_interactions[-1000:]
            
            # Save updated interactions
            with open(interactions_file, 'w', encoding='utf-8') as f:
                json.dump(existing_interactions, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving voice interaction: {str(e)}")
    
    def get_voice_navigation_commands(self, language: str = 'hindi') -> Dict:
        """Get available voice navigation commands"""
        try:
            nav_commands = self.voice_commands.get('navigation', {}).get(language, {})
            farming_commands = self.voice_commands.get('farming_actions', {}).get(language, {})
            
            return {
                'language': language,
                'navigation_commands': nav_commands,
                'farming_commands': farming_commands,
                'example_phrases': self._get_example_phrases(language),
                'voice_tips': self._get_voice_usage_tips(language)
            }
            
        except Exception as e:
            logger.error(f"Error getting voice commands: {str(e)}")
            return {}
    
    def _get_example_phrases(self, language: str) -> List[str]:
        """Get example phrases users can say"""
        examples = {
            'hindi': [
                "मुझे धान की खेती के बारे में बताओ",
                "आज का मौसम कैसा है दिल्ली में",
                "गेहूं का भाव क्या है आज",
                "मेरी जमीन के लिए कौन सी फसल अच्छी है",
                "खाद कब और कैसे डालना चाहिए",
                "फसल में कीड़े लगे हैं क्या करूं"
            ],
            'english': [
                "Tell me about rice farming",
                "What's the weather like in Delhi today",
                "What is the current wheat price",
                "Which crop is best for my land",
                "When and how should I apply fertilizer",
                "My crop has pest problem what to do"
            ],
            'punjabi': [
                "ਮੈਨੂੰ ਧਾਨ ਦੀ ਖੇਤੀ ਬਾਰੇ ਦੱਸੋ",
                "ਅੱਜ ਦਿੱਲੀ ਵਿੱਚ ਮੌਸਮ ਕਿਹੋ ਜਿਹਾ ਹੈ",
                "ਕਣਕ ਦਾ ਭਾਅ ਕੀ ਹੈ ਅੱਜ"
            ]
        }
        
        return examples.get(language, examples['english'])
    
    def _get_voice_usage_tips(self, language: str) -> List[str]:
        """Get tips for better voice interaction"""
        tips = {
            'hindi': [
                "📢 साफ और धीरे बोलें",
                "🎤 फोन माइक के पास बोलें",
                "🔇 शोर वाली जगह से बचें",
                "⏸️ बोलने के बाद थोड़ा इंतजार करें",
                "🔄 अगर समझ न आए तो दोबारा कोशिश करें"
            ],
            'english': [
                "📢 Speak clearly and slowly",
                "🎤 Speak close to the microphone",
                "🔇 Avoid noisy environments",
                "⏸️ Wait a moment after speaking",
                "🔄 Try again if not understood"
            ]
        }
        
        return tips.get(language, tips['english'])
    
    def get_voice_interaction_history(self, limit: int = 20) -> List[Dict]:
        """Get recent voice interaction history"""
        try:
            interactions_file = 'static/voice_interactions.json'
            
            if not os.path.exists(interactions_file):
                return []
            
            with open(interactions_file, 'r', encoding='utf-8') as f:
                interactions = json.load(f)
            
            # Return most recent interactions
            return interactions[-limit:]
            
        except Exception as e:
            logger.error(f"Error getting voice interaction history: {str(e)}")
            return []
    
    def analyze_voice_usage_patterns(self) -> Dict:
        """Analyze voice usage patterns for improvement"""
        try:
            interactions = self.get_voice_interaction_history(100)
            
            if not interactions:
                return {'error': 'No voice interaction data available'}
            
            # Analyze patterns
            language_usage = {}
            intent_distribution = {}
            confidence_scores = []
            
            for interaction in interactions:
                # Language usage
                lang = interaction['language']
                language_usage[lang] = language_usage.get(lang, 0) + 1
                
                # Intent distribution
                intent = interaction['intent']
                intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
                
                # Confidence scores
                confidence_scores.append(interaction['confidence_score'])
            
            # Calculate statistics
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            low_confidence_interactions = len([c for c in confidence_scores if c < 0.6])
            
            analysis = {
                'total_interactions': len(interactions),
                'language_distribution': language_usage,
                'intent_distribution': intent_distribution,
                'confidence_statistics': {
                    'average_confidence': round(avg_confidence, 2),
                    'low_confidence_count': low_confidence_interactions,
                    'success_rate': round((1 - low_confidence_interactions / len(interactions)) * 100, 1)
                },
                'most_popular_language': max(language_usage.items(), key=lambda x: x[1])[0],
                'most_common_intent': max(intent_distribution.items(), key=lambda x: x[1])[0],
                'improvement_suggestions': self._generate_voice_improvement_suggestions(avg_confidence, intent_distribution)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing voice usage patterns: {str(e)}")
            return {'error': str(e)}
    
    def _generate_voice_improvement_suggestions(self, avg_confidence: float, intent_distribution: Dict) -> List[str]:
        """Generate suggestions for improving voice system"""
        suggestions = []
        
        if avg_confidence < 0.7:
            suggestions.extend([
                "Improve speech recognition accuracy",
                "Add more training data for local accents",
                "Enhance noise cancellation"
            ])
        
        # Check for common intents that might need better responses
        common_intents = sorted(intent_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for intent, count in common_intents:
            if intent == 'general_inquiry':
                suggestions.append("Add more specific response templates for general queries")
            elif count > len(intent_distribution) * 0.3:  # Very common intent
                suggestions.append(f"Enhance responses for {intent} as it's frequently used")
        
        return suggestions

class VoiceAssistant:
    """High-level voice assistant for Smart Farmer application"""
    
    def __init__(self):
        self.voice_system = VoiceSupportSystem()
        self.conversation_context = {}
    
    def start_voice_session(self, user_id: str = None, language: str = 'hindi') -> Dict:
        """Start a new voice interaction session"""
        try:
            session_id = str(uuid.uuid4())
            
            # Initialize conversation context
            self.conversation_context[session_id] = {
                'user_id': user_id,
                'language': language,
                'start_time': datetime.now(),
                'interactions': [],
                'user_profile': {
                    'preferred_language': language,
                    'farming_interests': [],
                    'location': None
                }
            }
            
            # Generate welcome message
            welcome_message = self._generate_welcome_message(language)
            
            return {
                'session_id': session_id,
                'status': 'active',
                'language': language,
                'welcome_message': welcome_message,
                'voice_commands': self.voice_system.get_voice_navigation_commands(language),
                'supported_languages': list(self.voice_system.supported_languages.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error starting voice session: {str(e)}")
            raise
    
    def process_voice_message(self, session_id: str, audio_data: bytes) -> Dict:
        """Process voice message in ongoing session"""
        try:
            if session_id not in self.conversation_context:
                return {'error': 'Invalid session ID'}
            
            context = self.conversation_context[session_id]
            language = context['language']
            
            # Process voice input
            interaction = self.voice_system.process_voice_input(audio_data, language)
            
            # Add to conversation context
            context['interactions'].append({
                'text': interaction.recognized_text,
                'intent': interaction.intent,
                'response': interaction.response_text,
                'confidence': interaction.confidence_score,
                'timestamp': interaction.timestamp.isoformat()
            })
            
            # Update user profile based on interaction
            self._update_user_profile(context, interaction)
            
            return {
                'session_id': session_id,
                'interaction_id': interaction.interaction_id,
                'recognized_text': interaction.recognized_text,
                'intent': interaction.intent,
                'response_text': interaction.response_text,
                'audio_response_available': interaction.audio_response is not None,
                'confidence_score': interaction.confidence_score,
                'suggestions': self._get_follow_up_suggestions(interaction.intent, language),
                'timestamp': interaction.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing voice message: {str(e)}")
            return {'error': str(e)}
    
    def _generate_welcome_message(self, language: str) -> str:
        """Generate personalized welcome message"""
        messages = {
            'hindi': "नमस्ते! मैं स्मार्ट किसान आपका आवाज सहायक हूं। आप मुझसे फसल, मौसम, दाम और खेती की जानकारी के बारे में पूछ सकते हैं।",
            'english': "Hello! I'm Smart Farmer voice assistant. You can ask me about crops, weather, prices, and farming information.",
            'punjabi': "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਸਮਾਰਟ ਕਿਸਾਨ ਤੁਹਾਡਾ ਆਵਾਜ਼ ਸਹਾਇਕ ਹਾਂ। ਤੁਸੀਂ ਮੈਨੂੰ ਫਸਲ, ਮੌਸਮ ਅਤੇ ਖੇਤੀ ਬਾਰੇ ਪੁੱਛ ਸਕਦੇ ਹੋ।"
        }
        
        return messages.get(language, messages['english'])
    
    def _update_user_profile(self, context: Dict, interaction: VoiceInteraction) -> None:
        """Update user profile based on voice interactions"""
        try:
            profile = context['user_profile']
            
            # Extract and update farming interests
            recognized_crops = self._extract_crops_from_text(interaction.recognized_text, interaction.language)
            for crop in recognized_crops:
                if crop not in profile['farming_interests']:
                    profile['farming_interests'].append(crop)
            
            # Update location if mentioned
            recognized_locations = self._extract_locations_from_text(interaction.recognized_text)
            if recognized_locations and not profile['location']:
                profile['location'] = recognized_locations[0]
            
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
    
    def _extract_crops_from_text(self, text: str, language: str) -> List[str]:
        """Extract crop names from text"""
        crops = []
        crop_patterns = self.voice_system.speech_patterns.get('crop_names', {}).get(language, {})
        
        text_lower = text.lower()
        for local_name, english_name in crop_patterns.items():
            if local_name in text_lower:
                crops.append(english_name)
        
        return crops
    
    def _extract_locations_from_text(self, text: str) -> List[str]:
        """Extract location names from text"""
        locations = []
        common_places = ['delhi', 'mumbai', 'pune', 'lucknow', 'kanpur', 'amritsar', 'chandigarh']
        
        text_lower = text.lower()
        for place in common_places:
            if place in text_lower:
                locations.append(place)
        
        return locations
    
    def _get_follow_up_suggestions(self, intent: str, language: str) -> List[str]:
        """Get follow-up suggestions based on current intent"""
        suggestions = {
            'hindi': {
                'crop_recommendation': [
                    "मिट्टी की जांच के बारे में पूछें",
                    "मौसम की जानकारी लें",
                    "फसल की कीमत जानें"
                ],
                'weather_inquiry': [
                    "सिंचाई की सलाह लें",
                    "फसल की सुरक्षा के बारे में पूछें",
                    "आने वाले दिनों का मौसम जानें"
                ],
                'price_inquiry': [
                    "बेचने का सही समय जानें",
                    "गुणवत्ता सुधार के तरीके पूछें",
                    "भंडारण की सलाह लें"
                ]
            },
            'english': [
                "Ask about soil testing",
                "Get weather information",
                "Check crop prices",
                "Learn about fertilizers",
                "Get pest control advice"
            ]
        }
        
        if language in suggestions and intent in suggestions[language]:
            return suggestions[language][intent]
        else:
            return suggestions.get('english', [])
    
    def end_voice_session(self, session_id: str) -> Dict:
        """End voice interaction session"""
        try:
            if session_id not in self.conversation_context:
                return {'error': 'Invalid session ID'}
            
            context = self.conversation_context[session_id]
            
            # Generate session summary
            session_summary = {
                'session_id': session_id,
                'duration_minutes': (datetime.now() - context['start_time']).total_seconds() / 60,
                'total_interactions': len(context['interactions']),
                'language_used': context['language'],
                'topics_discussed': list(set(i['intent'] for i in context['interactions'])),
                'user_profile_updates': context['user_profile'],
                'session_quality': self._assess_session_quality(context),
                'end_time': datetime.now().isoformat()
            }
            
            # Save session summary
            self._save_session_summary(session_summary)
            
            # Clean up session context
            del self.conversation_context[session_id]
            
            return {
                'status': 'session_ended',
                'summary': session_summary,
                'farewell_message': self._generate_farewell_message(context['language'])
            }
            
        except Exception as e:
            logger.error(f"Error ending voice session: {str(e)}")
            return {'error': str(e)}
    
    def _assess_session_quality(self, context: Dict) -> Dict:
        """Assess the quality of voice interaction session"""
        interactions = context['interactions']
        
        if not interactions:
            return {'quality': 'no_data'}
        
        avg_confidence = sum(i['confidence'] for i in interactions) / len(interactions)
        successful_interactions = len([i for i in interactions if i['confidence'] > 0.7])
        
        quality_score = (avg_confidence + (successful_interactions / len(interactions))) / 2
        
        return {
            'quality_score': round(quality_score, 2),
            'average_confidence': round(avg_confidence, 2),
            'success_rate': round((successful_interactions / len(interactions)) * 100, 1),
            'total_interactions': len(interactions),
            'quality_rating': 'excellent' if quality_score > 0.8 else 'good' if quality_score > 0.6 else 'needs_improvement'
        }
    
    def _generate_farewell_message(self, language: str) -> str:
        """Generate farewell message"""
        messages = {
            'hindi': "धन्यवाद! खुशी से मदद की। खेती में कोई भी समस्या हो तो वापस आइए। शुभ दिन!",
            'english': "Thank you! Happy to help. Come back anytime for farming advice. Have a great day!",
            'punjabi': "ਧੰਨਵਾਦ! ਮਦਦ ਕਰਕੇ ਖੁਸ਼ੀ ਹੋਈ। ਖੇਤੀ ਵਿੱਚ ਕੋਈ ਸਮੱਸਿਆ ਹੋਵੇ ਤਾਂ ਵਾਪਸ ਆਉਣਾ।"
        }
        
        return messages.get(language, messages['english'])
    
    def _save_session_summary(self, summary: Dict) -> None:
        """Save session summary for analytics"""
        try:
            sessions_file = 'static/voice_sessions.json'
            
            # Load existing sessions
            if os.path.exists(sessions_file):
                with open(sessions_file, 'r', encoding='utf-8') as f:
                    existing_sessions = json.load(f)
            else:
                existing_sessions = []
            
            # Add new session
            existing_sessions.append(summary)
            
            # Keep only last 500 sessions
            if len(existing_sessions) > 500:
                existing_sessions = existing_sessions[-500:]
            
            # Save updated sessions
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(existing_sessions, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving session summary: {str(e)}")

# Global voice assistant instance
voice_assistant = VoiceAssistant()

def create_voice_session(user_id: str = None, language: str = 'hindi') -> Dict:
    """Create new voice interaction session"""
    try:
        return voice_assistant.start_voice_session(user_id, language)
    except Exception as e:
        logger.error(f"Error creating voice session: {str(e)}")
        return {'error': str(e)}

def process_voice_input(session_id: str, audio_base64: str) -> Dict:
    """Process voice input from user"""
    try:
        # Decode audio data
        audio_data = base64.b64decode(audio_base64)
        
        return voice_assistant.process_voice_message(session_id, audio_data)
        
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        return {'error': str(e)}

def get_voice_capabilities() -> Dict:
    """Get voice system capabilities and supported features"""
    try:
        return {
            'supported_languages': list(voice_assistant.voice_system.supported_languages.keys()),
            'language_details': voice_assistant.voice_system.supported_languages,
            'supported_intents': [
                'crop_recommendation',
                'weather_inquiry', 
                'price_inquiry',
                'fertilizer_advice',
                'disease_pest_help',
                'yield_prediction',
                'general_inquiry'
            ],
            'voice_features': [
                'Speech-to-Text Recognition',
                'Text-to-Speech Response',
                'Multilingual Support',
                'Agricultural Terminology Recognition',
                'Voice Navigation Commands',
                'Conversation Context Management'
            ],
            'accessibility_features': [
                'Slow speech support',
                'Repeat functionality',
                'Simple language responses',
                'Voice command shortcuts',
                'Audio cues and feedback'
            ],
            'usage_tips': voice_assistant.voice_system._get_voice_usage_tips('english'),
            'sample_commands': voice_assistant.voice_system._get_example_phrases('english')
        }
        
    except Exception as e:
        logger.error(f"Error getting voice capabilities: {str(e)}")
        return {'error': str(e)}

def get_voice_analytics() -> Dict:
    """Get voice system usage analytics"""
    try:
        return voice_assistant.voice_system.analyze_voice_usage_patterns()
    except Exception as e:
        logger.error(f"Error getting voice analytics: {str(e)}")
        return {'error': str(e)}

def convert_text_to_voice(text: str, language: str = 'hindi') -> Dict:
    """Convert text to voice for testing purposes"""
    try:
        # This would integrate with actual TTS services
        voice_system = VoiceSupportSystem()
        audio_data = voice_system._text_to_speech(text, language)
        
        # Encode audio data to base64 for transmission
        audio_base64 = base64.b64encode(audio_data).decode('utf-8') if audio_data else ""
        
        return {
            'text': text,
            'language': language,
            'audio_available': len(audio_base64) > 0,
            'audio_base64': audio_base64,
            'voice_used': voice_system.supported_languages.get(language, {}).get('tts_voice', 'default'),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error converting text to voice: {str(e)}")
        return {'error': str(e)}

# Accessibility helper functions
def create_simplified_voice_interface(language: str = 'hindi') -> Dict:
    """Create simplified voice interface for low-literate users"""
    try:
        # Simple, limited-option interface
        simple_commands = {
            'hindi': {
                'options': [
                    {'command': 'फसल', 'action': 'crop_recommendation', 'description': 'फसल की सलाह'},
                    {'command': 'मौसम', 'action': 'weather', 'description': 'मौसम की जानकारी'},
                    {'command': 'दाम', 'action': 'prices', 'description': 'फसल के दाम'},
                    {'command': 'मदद', 'action': 'help', 'description': 'सहायता और मार्गदर्शन'}
                ],
                'instructions': [
                    "बटन दबाकर बोलें",
                    "साफ और धीरे बोलें",
                    "एक समय में एक ही सवाल पूछें",
                    "जवाब सुनने के बाद अगला सवाल पूछें"
                ]
            },
            'english': {
                'options': [
                    {'command': 'crop', 'action': 'crop_recommendation', 'description': 'Crop advice'},
                    {'command': 'weather', 'action': 'weather', 'description': 'Weather information'},
                    {'command': 'price', 'action': 'prices', 'description': 'Crop prices'},
                    {'command': 'help', 'action': 'help', 'description': 'Help and guidance'}
                ],
                'instructions': [
                    "Press button and speak",
                    "Speak clearly and slowly",
                    "Ask one question at a time",
                    "Wait for response before next question"
                ]
            }
        }
        
        return {
            'interface_type': 'simplified',
            'language': language,
            'voice_options': simple_commands.get(language, simple_commands['english']),
            'accessibility_features': [
                'Large buttons for easy touch',
                'Visual feedback for voice recognition',
                'Automatic speech replay option',
                'Simple yes/no confirmations'
            ],
            'setup_instructions': [
                'Ensure microphone permission is granted',
                'Use in quiet environment for best results',
                'Keep phone close to mouth while speaking',
                'Speak in local language for better recognition'
            ]
        }
        
    except Exception as e:
        logger.error(f"Error creating simplified interface: {str(e)}")
        return {'error': str(e)}

def get_voice_training_data() -> Dict:
    """Get voice training data for improving recognition"""
    try:
        # This would help improve the voice recognition system
        training_data = {
            'common_mispronunciations': {
                'hindi': {
                    'धान': ['धन', 'धन्न', 'धाण'],
                    'गेहूं': ['गेहु', 'गहूं', 'गेहुँ'],
                    'कपास': ['कपाश', 'कपस', 'कप्पास']
                }
            },
            'regional_variations': {
                'wheat': {
                    'hindi': ['गेहूं', 'गेहुँ'],
                    'punjabi': ['ਕਣਕ'],
                    'haryanvi': ['गूंम'],
                    'rajasthani': ['गेंहूं']
                }
            },
            'context_patterns': [
                'Price inquiries often include time references',
                'Crop questions usually include location',
                'Weather queries often mention specific activities'
            ],
            'improvement_areas': [
                'Better recognition of numerical values',
                'Improved location name recognition',
                'Enhanced agricultural terminology'
            ]
        }
        
        return training_data
        
    except Exception as e:
        logger.error(f"Error getting voice training data: {str(e)}")
        return {'error': str(e)}

import uuid
