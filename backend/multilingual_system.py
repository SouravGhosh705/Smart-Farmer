#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset-based Multilingual Translation System
Provides translations using local CSV datasets instead of APIs
"""

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class MultilingualSystem:
    """Handle multilingual translations using local datasets"""
    
    def __init__(self):
        self.crops_df = None
        self.diseases_df = None
        self.symptoms_df = None
        self.fertilizers_df = None
        self.ui_df = None
        self.supported_languages = ['english', 'hindi', 'gujarati', 'punjabi', 'marathi', 'tamil', 'telugu', 'bengali']
        self.load_datasets()
    
    def load_datasets(self):
        """Load all multilingual datasets"""
        try:
            base_path = 'static/datasets/'
            
            # Load crop translations
            self.crops_df = pd.read_csv(f'{base_path}multilingual_crops.csv')
            logger.info("Loaded multilingual crops dataset")
            
            # Load disease translations
            self.diseases_df = pd.read_csv(f'{base_path}multilingual_diseases.csv')
            logger.info("Loaded multilingual diseases dataset")
            
            # Load symptom translations
            self.symptoms_df = pd.read_csv(f'{base_path}multilingual_symptoms.csv')
            logger.info("Loaded multilingual symptoms dataset")
            
            # Load fertilizer translations
            self.fertilizers_df = pd.read_csv(f'{base_path}multilingual_fertilizers.csv')
            logger.info("Loaded multilingual fertilizers dataset")
            
            # Try to load comprehensive UI translations first, fallback to basic
            try:
                self.ui_df = pd.read_csv(f'{base_path}comprehensive_ui_translations.csv')
                logger.info("Loaded comprehensive UI translations dataset")
            except:
                self.ui_df = pd.read_csv(f'{base_path}multilingual_ui.csv')
                logger.info("Loaded basic multilingual UI dataset")
            
            return True
        except Exception as e:
            logger.error(f"Error loading multilingual datasets: {str(e)}")
            return False
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        return [
            {'code': 'en', 'name': 'English', 'native': 'English'},
            {'code': 'hi', 'name': 'Hindi', 'native': 'हिन्दी'},
            {'code': 'gu', 'name': 'Gujarati', 'native': 'ગુજરાતી'},
            {'code': 'pa', 'name': 'Punjabi', 'native': 'ਪੰਜਾਬੀ'},
            {'code': 'mr', 'name': 'Marathi', 'native': 'मराठी'},
            {'code': 'ta', 'name': 'Tamil', 'native': 'தமிழ்'},
            {'code': 'te', 'name': 'Telugu', 'native': 'తెలుగు'},
            {'code': 'bn', 'name': 'Bengali', 'native': 'বাংলা'}
        ]
    
    def translate_crop(self, crop_name, target_language='english'):
        """Translate crop name to target language"""
        if self.crops_df is None:
            return crop_name
        
        # Find crop in English
        crop_row = self.crops_df[self.crops_df['crop_english'].str.lower() == crop_name.lower()]
        
        if crop_row.empty:
            return crop_name
        
        # Get translation column name
        target_col = f'crop_{target_language.lower()}'
        
        if target_col not in self.crops_df.columns:
            return crop_name
        
        return crop_row.iloc[0][target_col]
    
    def translate_disease(self, disease_name, target_language='english'):
        """Translate disease name to target language"""
        if self.diseases_df is None:
            return disease_name
        
        disease_row = self.diseases_df[self.diseases_df['disease_english'].str.lower() == disease_name.lower()]
        
        if disease_row.empty:
            return disease_name
        
        target_col = f'disease_{target_language.lower()}'
        
        if target_col not in self.diseases_df.columns:
            return disease_name
        
        return disease_row.iloc[0][target_col]
    
    def translate_symptom(self, symptom_name, target_language='english'):
        """Translate symptom name to target language"""
        if self.symptoms_df is None:
            return symptom_name
        
        symptom_row = self.symptoms_df[self.symptoms_df['symptom_english'].str.lower() == symptom_name.lower()]
        
        if symptom_row.empty:
            return symptom_name
        
        target_col = f'symptom_{target_language.lower()}'
        
        if target_col not in self.symptoms_df.columns:
            return symptom_name
        
        return symptom_row.iloc[0][target_col]
    
    def translate_fertilizer(self, fertilizer_name, target_language='english'):
        """Translate fertilizer name to target language"""
        if self.fertilizers_df is None:
            return fertilizer_name
        
        fertilizer_row = self.fertilizers_df[self.fertilizers_df['fertilizer_english'].str.lower() == fertilizer_name.lower()]
        
        if fertilizer_row.empty:
            return fertilizer_name
        
        target_col = f'fertilizer_{target_language.lower()}'
        
        if target_col not in self.fertilizers_df.columns:
            return fertilizer_name
        
        return fertilizer_row.iloc[0][target_col]
    
    def translate_ui_text(self, key, target_language='english'):
        """Translate UI text to target language"""
        if self.ui_df is None:
            return key
        
        ui_row = self.ui_df[self.ui_df['key'] == key]
        
        if ui_row.empty:
            # Return fallback for crop doctor specific keys
            crop_doctor_fallbacks = {
                'title': {
                    'hindi': 'AI फसल डॉक्टर',
                    'gujarati': 'AI પાક ડૉક્ટર',
                    'punjabi': 'AI ਫਸਲ ਡਾਕਟਰ',
                    'marathi': 'AI पीक डॉक्टर',
                    'tamil': 'AI பயிர் மருத்துவர்',
                    'telugu': 'AI పంట వైద్యుడు',
                    'bengali': 'AI ফসল ডাক্তার'
                },
                'subtitle': {
                    'hindi': 'AI-संचालित रोग और कीट की पहचान के लिए पौधों की छवियां अपलोड करें',
                    'gujarati': 'AI-સંચાલિત રોગ અને કીટ ઓળખ માટે છોડની છબીઓ અપલોડ કરો',
                    'punjabi': 'AI-ਸੰਚਾਲਿਤ ਬਿਮਾਰੀ ਅਤੇ ਕੀੜੇ ਦੀ ਪਛਾਣ ਲਈ ਪੌਧਿਆਂ ਦੀਆਂ ਤਸਵੀਰਾਂ ਅੱਪਲੋਡ ਕਰੋ',
                    'marathi': 'AI-चालित रोग आणि किटक ओळखण्यासाठी वनस्पती प्रतिमा अपलोड करा',
                    'tamil': 'AI-இயக்கப்படும் நோய் மற்றும் பூச்சி அடையாளத்திற்காக தாவர படங்களை பதிவேற்றவும்',
                    'telugu': 'AI-శక్తితో నడిచే వ్యాధి మరియు చీడపురుగుల గుర్తింపు కోసం మొక్కల చిత్రాలను అప్‌లోడ్ చేయండి',
                    'bengali': 'AI-চালিত রোগ এবং পোকামাকড় সনাক্তকরণের জন্য উদ্ভিদের ছবি আপলোড করুন'
                },
                'analyzeButton': {
                    'hindi': 'पौधे की स्वास्थ्य का विश्लेषण करें',
                    'gujarati': 'છોડના આરોગ્યનું વિશ્લેષણ કરો',
                    'punjabi': 'ਪੌਧੇ ਦੀ ਸਿਹਤ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰੋ',
                    'marathi': 'वनस्पती आरोग्याचे विश्लेषण करा',
                    'tamil': 'தாவர ஆரோக்கியத்தை ஆய்வு செய்யுங்கள்',
                    'telugu': 'మొక్క ఆరోగ్యాన్ని విశ్లేషించండి',
                    'bengali': 'উদ্ভিদের স্বাস্থ্য বিশ্লেষণ করুন'
                },
                'severity': {
                    'hindi': 'गंभीरता',
                    'gujarati': 'ગંભીરતા',
                    'punjabi': 'ਗੰਭੀਰਤਾ',
                    'marathi': 'तीव्रता',
                    'tamil': 'தீவிரம்',
                    'telugu': 'తీవ్రత',
                    'bengali': 'তীব্রতা'
                },
                'recommendations': {
                    'hindi': 'उपचार की सिफारिशें',
                    'gujarati': 'સારવારની ભલામણો',
                    'punjabi': 'ਇਲਾਜ ਦੀਆਂ ਸਿਫਾਰਿਸ਼ਾਂ',
                    'marathi': 'उपचार शिफारसी',
                    'tamil': 'சிகிச்சை பரிந்துரைகள்',
                    'telugu': 'చికిత్స సిఫార్సులు',
                    'bengali': 'চিকিৎসার সুপারিশ'
                }
            }
            
            # Check if key exists in fallbacks
            if key in crop_doctor_fallbacks and target_language.lower() in crop_doctor_fallbacks[key]:
                return crop_doctor_fallbacks[key][target_language.lower()]
            
            return key
        
        if target_language.lower() not in self.ui_df.columns:
            return ui_row.iloc[0]['english']
        
        return ui_row.iloc[0][target_language.lower()]
    
    def translate_crop_list(self, crop_list, target_language='english'):
        """Translate a list of crop recommendations with confidence scores"""
        translated_list = []
        
        for crop_data in crop_list:
            if isinstance(crop_data, list) and len(crop_data) == 2:
                crop_name, confidence = crop_data
                translated_crop = self.translate_crop(crop_name, target_language)
                translated_list.append([translated_crop, confidence])
            else:
                translated_list.append(crop_data)
        
        return translated_list
    
    def translate_response(self, response_data, target_language='english'):
        """Translate entire API response to target language"""
        if target_language.lower() == 'english' or target_language.lower() == 'en':
            return response_data
        
        translated_response = response_data.copy()
        
        # Translate crop recommendations
        if 'crop' in translated_response:
            translated_response['crop'] = self.translate_crop(translated_response['crop'], target_language)
        
        if 'crop_list' in translated_response:
            translated_response['crop_list'] = self.translate_crop_list(translated_response['crop_list'], target_language)
        
        # Add language metadata
        translated_response['language'] = target_language
        translated_response['supported_languages'] = self.get_supported_languages()
        
        return translated_response
    
    def detect_language_from_text(self, text):
        """Simple language detection based on character sets"""
        if not text:
            return 'english'
        
        # Check for specific language characters
        if any('\u0900' <= char <= '\u097F' for char in text):  # Devanagari (Hindi/Marathi)
            if any(char in 'ळ' for char in text):
                return 'marathi'
            return 'hindi'
        elif any('\u0A80' <= char <= '\u0AFF' for char in text):  # Gujarati
            return 'gujarati'
        elif any('\u0A00' <= char <= '\u0A7F' for char in text):  # Punjabi
            return 'punjabi'
        elif any('\u0B80' <= char <= '\u0BFF' for char in text):  # Tamil
            return 'tamil'
        elif any('\u0C00' <= char <= '\u0C7F' for char in text):  # Telugu
            return 'telugu'
        elif any('\u0980' <= char <= '\u09FF' for char in text):  # Bengali
            return 'bengali'
        else:
            return 'english'

# Initialize global multilingual system
multilingual_system = MultilingualSystem()

def get_multilingual_system():
    """Get the global multilingual system instance"""
    return multilingual_system
