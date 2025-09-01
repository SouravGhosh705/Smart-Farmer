#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pest/Disease Detection Module for Smart Farmer Application
AI-based plant disease and pest detection with image upload functionality
"""

import os
import cv2
import numpy as np
import base64
import io
from PIL import Image
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import uuid
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Data class for detection results"""
    detection_id: str
    image_path: str
    detected_issues: List[Dict]
    confidence_scores: List[float]
    treatment_recommendations: List[str]
    severity: str
    crop_type: str
    timestamp: datetime

class PlantDiseaseDetector:
    """AI-based plant disease and pest detection system"""
    
    def __init__(self):
        self.disease_database = self._load_disease_database()
        self.pest_database = self._load_pest_database()
        self.treatment_database = self._load_treatment_database()
        os.makedirs('static/uploads', exist_ok=True)
        os.makedirs('static/disease_images', exist_ok=True)
    
    def _load_disease_database(self) -> Dict:
        """Load comprehensive plant disease database"""
        return {
            'rice': {
                'blast': {
                    'symptoms': ['oval or boat-shaped lesions', 'gray center with brown border', 'leaf death'],
                    'causes': ['Magnaporthe oryzae fungus', 'high humidity', 'nitrogen excess'],
                    'favorable_conditions': ['temperature 25-28°C', 'humidity >90%', 'cloudy weather'],
                    'severity_indicators': {
                        'mild': 'few lesions on lower leaves',
                        'moderate': 'lesions on multiple leaves',
                        'severe': 'neck blast, panicle infection'
                    }
                },
                'brown_spot': {
                    'symptoms': ['small brown spots', 'yellow halos', 'premature leaf death'],
                    'causes': ['Bipolaris oryzae', 'potassium deficiency', 'drought stress'],
                    'favorable_conditions': ['temperature 25-30°C', 'intermittent rain'],
                    'severity_indicators': {
                        'mild': 'few spots on older leaves',
                        'moderate': 'spots spreading to younger leaves',
                        'severe': 'severe defoliation'
                    }
                },
                'bacterial_blight': {
                    'symptoms': ['water-soaked lesions', 'yellow stripes', 'wilting'],
                    'causes': ['Xanthomonas oryzae', 'wounds', 'flooding'],
                    'favorable_conditions': ['temperature 25-30°C', 'high humidity', 'rain'],
                    'severity_indicators': {
                        'mild': 'few lesions on leaf tips',
                        'moderate': 'lesions extending into leaf blade',
                        'severe': 'complete leaf death, systemic infection'
                    }
                }
            },
            'wheat': {
                'rust': {
                    'symptoms': ['orange/brown pustules', 'powdery spores', 'yellowing'],
                    'causes': ['Puccinia species', 'moisture', 'moderate temperatures'],
                    'favorable_conditions': ['temperature 15-25°C', 'high humidity', 'dew'],
                    'severity_indicators': {
                        'mild': 'few pustules on lower leaves',
                        'moderate': 'pustules on flag leaf',
                        'severe': 'severe defoliation, yield loss'
                    }
                },
                'smut': {
                    'symptoms': ['black powdery masses', 'destroyed grains', 'fishy odor'],
                    'causes': ['Tilletia species', 'infected seeds', 'soil contamination'],
                    'favorable_conditions': ['temperature 16-22°C', 'moist soil'],
                    'severity_indicators': {
                        'mild': 'few affected tillers',
                        'moderate': '10-25% tillers affected',
                        'severe': '>25% tillers affected'
                    }
                }
            },
            'cotton': {
                'bollworm': {
                    'symptoms': ['holes in bolls', 'damaged squares', 'frass presence'],
                    'causes': ['Helicoverpa armigera', 'warm weather', 'flowering stage'],
                    'favorable_conditions': ['temperature 25-30°C', 'dry weather'],
                    'severity_indicators': {
                        'mild': '<5% boll damage',
                        'moderate': '5-15% boll damage',
                        'severe': '>15% boll damage'
                    }
                },
                'wilt': {
                    'symptoms': ['yellowing leaves', 'wilting', 'vascular browning'],
                    'causes': ['Fusarium oxysporum', 'soil-borne pathogen', 'water stress'],
                    'favorable_conditions': ['temperature 25-30°C', 'alkaline soil'],
                    'severity_indicators': {
                        'mild': 'few plants affected',
                        'moderate': 'patches of affected plants',
                        'severe': 'widespread wilting'
                    }
                }
            },
            'tomato': {
                'early_blight': {
                    'symptoms': ['dark spots with concentric rings', 'yellowing', 'defoliation'],
                    'causes': ['Alternaria solani', 'warm humid conditions'],
                    'favorable_conditions': ['temperature 24-29°C', 'humidity >90%'],
                    'severity_indicators': {
                        'mild': 'few spots on lower leaves',
                        'moderate': 'spreading to upper leaves',
                        'severe': 'severe defoliation, fruit infection'
                    }
                },
                'late_blight': {
                    'symptoms': ['water-soaked lesions', 'white fuzzy growth', 'rapid spread'],
                    'causes': ['Phytophthora infestans', 'cool moist conditions'],
                    'favorable_conditions': ['temperature 18-22°C', 'humidity >95%'],
                    'severity_indicators': {
                        'mild': 'few lesions on leaves',
                        'moderate': 'stem and fruit infection',
                        'severe': 'plant death within days'
                    }
                }
            }
        }
    
    def _load_pest_database(self) -> Dict:
        """Load comprehensive pest database"""
        return {
            'aphids': {
                'appearance': ['small soft-bodied insects', 'green/black color', 'cluster on leaves'],
                'damage': ['leaf curling', 'stunted growth', 'honeydew secretion'],
                'crops_affected': ['wheat', 'cotton', 'vegetables', 'fruit_trees'],
                'life_cycle': '7-10 days',
                'peak_season': 'winter and spring'
            },
            'whitefly': {
                'appearance': ['tiny white flying insects', 'found on leaf undersides'],
                'damage': ['yellowing leaves', 'stunted growth', 'virus transmission'],
                'crops_affected': ['cotton', 'tomato', 'chili', 'eggplant'],
                'life_cycle': '15-30 days',
                'peak_season': 'warm weather'
            },
            'thrips': {
                'appearance': ['tiny slender insects', 'yellow to brown color'],
                'damage': ['silvery streaks on leaves', 'black specks', 'distorted growth'],
                'crops_affected': ['cotton', 'vegetables', 'flowers'],
                'life_cycle': '14-21 days',
                'peak_season': 'hot dry weather'
            },
            'bollworm': {
                'appearance': ['caterpillars', 'greenish to brown color', '25-40mm length'],
                'damage': ['holes in fruits/bolls', 'damaged squares'],
                'crops_affected': ['cotton', 'tomato', 'chickpea', 'pigeon_pea'],
                'life_cycle': '25-35 days',
                'peak_season': 'monsoon and post-monsoon'
            },
            'stem_borer': {
                'appearance': ['yellowish caterpillars', 'bore into stems'],
                'damage': ['dead hearts', 'white ears', 'stem tunneling'],
                'crops_affected': ['rice', 'sugarcane', 'maize'],
                'life_cycle': '25-45 days',
                'peak_season': 'kharif season'
            },
            'leaf_miner': {
                'appearance': ['tiny flies', 'serpentine mines in leaves'],
                'damage': ['mining patterns', 'reduced photosynthesis'],
                'crops_affected': ['vegetables', 'citrus', 'ornamentals'],
                'life_cycle': '21-28 days',
                'peak_season': 'warm humid conditions'
            }
        }
    
    def _load_treatment_database(self) -> Dict:
        """Load treatment and management strategies"""
        return {
            'organic_treatments': {
                'neem_oil': {
                    'active_ingredient': 'Azadirachtin',
                    'effective_against': ['aphids', 'whitefly', 'thrips', 'leaf_miner'],
                    'application': '2-3ml per liter water, spray in evening',
                    'safety_period': '3 days before harvest'
                },
                'bt_spray': {
                    'active_ingredient': 'Bacillus thuringiensis',
                    'effective_against': ['bollworm', 'stem_borer', 'caterpillars'],
                    'application': '1-2gm per liter water',
                    'safety_period': '1 day before harvest'
                },
                'garlic_chili_spray': {
                    'active_ingredient': 'Natural deterrents',
                    'effective_against': ['aphids', 'small insects'],
                    'application': 'Homemade spray, safe for organic farming',
                    'safety_period': 'No waiting period'
                },
                'trichoderma': {
                    'active_ingredient': 'Beneficial fungus',
                    'effective_against': ['soil-borne diseases', 'root rot', 'wilt'],
                    'application': 'Soil application or seed treatment',
                    'safety_period': 'No waiting period'
                }
            },
            'chemical_treatments': {
                'imidacloprid': {
                    'active_ingredient': 'Neonicotinoid',
                    'effective_against': ['aphids', 'whitefly', 'thrips'],
                    'application': '0.5ml per liter water',
                    'safety_period': '21 days before harvest'
                },
                'chlorpyrifos': {
                    'active_ingredient': 'Organophosphate',
                    'effective_against': ['bollworm', 'stem_borer'],
                    'application': '2ml per liter water',
                    'safety_period': '15 days before harvest'
                },
                'mancozeb': {
                    'active_ingredient': 'Fungicide',
                    'effective_against': ['fungal diseases', 'blight', 'rust'],
                    'application': '2gm per liter water',
                    'safety_period': '7 days before harvest'
                }
            },
            'biological_control': {
                'ladybird_beetles': {
                    'target_pests': ['aphids', 'scale_insects'],
                    'release_rate': '100-200 per hectare',
                    'establishment_time': '2-3 weeks'
                },
                'predatory_mites': {
                    'target_pests': ['spider_mites', 'thrips'],
                    'release_rate': '50-100 per square meter',
                    'establishment_time': '1-2 weeks'
                },
                'trichogramma': {
                    'target_pests': ['bollworm', 'stem_borer'],
                    'release_rate': '1-2 lakh per hectare',
                    'establishment_time': '1 week'
                }
            },
            'cultural_practices': {
                'crop_rotation': {
                    'benefits': ['break pest cycles', 'reduce soil-borne diseases'],
                    'implementation': 'Rotate with non-host crops annually'
                },
                'clean_cultivation': {
                    'benefits': ['remove pest habitats', 'reduce disease inoculum'],
                    'implementation': 'Remove crop residues, weeds regularly'
                },
                'resistant_varieties': {
                    'benefits': ['natural pest resistance', 'reduced chemical use'],
                    'implementation': 'Choose certified resistant varieties'
                }
            }
        }
    
    def analyze_plant_image(self, image_data: bytes, crop_type: str = None) -> DetectionResult:
        """Analyze plant image for diseases and pests"""
        try:
            # Generate unique ID for this detection
            detection_id = str(uuid.uuid4())
            
            # Save uploaded image
            image_path = f"static/uploads/plant_image_{detection_id}.jpg"
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Invalid image format")
            
            # Perform image analysis
            analysis_results = self._perform_image_analysis(image, crop_type)
            
            # Generate treatment recommendations
            recommendations = self._generate_treatment_recommendations(
                analysis_results['detected_issues'], 
                crop_type
            )
            
            # Determine overall severity
            severity = self._determine_severity(analysis_results['detected_issues'])
            
            result = DetectionResult(
                detection_id=detection_id,
                image_path=image_path,
                detected_issues=analysis_results['detected_issues'],
                confidence_scores=analysis_results['confidence_scores'],
                treatment_recommendations=recommendations,
                severity=severity,
                crop_type=crop_type or 'unknown',
                timestamp=datetime.now()
            )
            
            # Save detection result
            self._save_detection_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing plant image: {str(e)}")
            raise
    
    def _perform_image_analysis(self, image: np.ndarray, crop_type: str = None) -> Dict:
        """Perform computer vision analysis on plant image"""
        try:
            # Basic image preprocessing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Image quality assessment
            quality_score = self._assess_image_quality(image)
            
            if quality_score < 0.5:
                return {
                    'detected_issues': [{
                        'type': 'image_quality',
                        'name': 'Poor Image Quality',
                        'description': 'Image quality is too poor for accurate analysis',
                        'confidence': 0.9
                    }],
                    'confidence_scores': [0.9],
                    'analysis_notes': 'Please upload a clearer image'
                }
            
            # Color analysis for disease detection
            color_analysis = self._analyze_color_patterns(image_rgb)
            
            # Texture analysis for disease/pest detection
            texture_analysis = self._analyze_texture_patterns(image)
            
            # Shape analysis for pest detection
            shape_analysis = self._analyze_shape_patterns(image)
            
            # Combine analyses to detect issues
            detected_issues = []
            confidence_scores = []
            
            # Disease detection based on color patterns
            disease_results = self._detect_diseases_from_color(color_analysis, crop_type)
            detected_issues.extend(disease_results['issues'])
            confidence_scores.extend(disease_results['confidences'])
            
            # Pest detection based on shape and texture
            pest_results = self._detect_pests_from_features(shape_analysis, texture_analysis, crop_type)
            detected_issues.extend(pest_results['issues'])
            confidence_scores.extend(pest_results['confidences'])
            
            # If no specific issues detected, provide general assessment
            if not detected_issues:
                general_assessment = self._general_plant_assessment(color_analysis, crop_type)
                detected_issues.append(general_assessment['assessment'])
                confidence_scores.append(general_assessment['confidence'])
            
            return {
                'detected_issues': detected_issues,
                'confidence_scores': confidence_scores,
                'analysis_details': {
                    'image_quality': quality_score,
                    'color_analysis': color_analysis,
                    'texture_analysis': texture_analysis,
                    'shape_analysis': shape_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            raise
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess image quality for analysis"""
        # Calculate image sharpness using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (higher = sharper)
        sharpness_score = min(1.0, laplacian_var / 500)
        
        # Check brightness
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128
        
        # Check contrast
        contrast_score = np.std(gray) / 128
        contrast_score = min(1.0, contrast_score)
        
        # Overall quality score
        quality_score = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
        
        return quality_score
    
    def _analyze_color_patterns(self, image_rgb: np.ndarray) -> Dict:
        """Analyze color patterns in the image"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Analyze green health (healthy plant indicator)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_percentage = np.sum(green_mask > 0) / (image_rgb.shape[0] * image_rgb.shape[1]) * 100
        
        # Analyze brown/yellow areas (disease indicators)
        brown_yellow_mask = cv2.inRange(hsv, (10, 40, 40), (35, 255, 255))
        brown_yellow_percentage = np.sum(brown_yellow_mask > 0) / (image_rgb.shape[0] * image_rgb.shape[1]) * 100
        
        # Analyze dark spots (disease indicators)
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
        dark_percentage = np.sum(dark_mask > 0) / (image_rgb.shape[0] * image_rgb.shape[1]) * 100
        
        # Analyze white areas (fungal growth indicators)
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        white_percentage = np.sum(white_mask > 0) / (image_rgb.shape[0] * image_rgb.shape[1]) * 100
        
        return {
            'green_health_percentage': round(green_percentage, 2),
            'brown_yellow_percentage': round(brown_yellow_percentage, 2),
            'dark_spots_percentage': round(dark_percentage, 2),
            'white_areas_percentage': round(white_percentage, 2),
            'overall_health_indicator': 'healthy' if green_percentage > 60 else 'stressed' if green_percentage > 30 else 'severely_affected'
        }
    
    def _analyze_texture_patterns(self, image: np.ndarray) -> Dict:
        """Analyze texture patterns for disease detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features using Local Binary Pattern simulation
        # Simplified texture analysis
        
        # Edge detection for spot/lesion boundaries
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1]) * 100
        
        # Texture variance
        texture_variance = np.var(gray)
        
        # Spot detection using contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spot_count = len([c for c in contours if cv2.contourArea(c) > 100])
        
        return {
            'edge_density': round(edge_density, 2),
            'texture_variance': round(texture_variance, 2),
            'detected_spots': spot_count,
            'texture_uniformity': 'uniform' if texture_variance < 1000 else 'variable'
        }
    
    def _analyze_shape_patterns(self, image: np.ndarray) -> Dict:
        """Analyze shape patterns for pest detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blob detection for pests
        # Simplified blob detection
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours for potential pests
        potential_pests = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 5000:  # Size range for visible pests
                # Calculate shape descriptors
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    potential_pests.append({
                        'area': area,
                        'circularity': circularity,
                        'aspect_ratio': aspect_ratio,
                        'location': (x, y, w, h)
                    })
        
        return {
            'potential_pest_count': len(potential_pests),
            'pest_features': potential_pests[:10],  # Limit to top 10
            'shape_analysis_complete': True
        }
    
    def _detect_diseases_from_color(self, color_analysis: Dict, crop_type: str = None) -> Dict:
        """Detect diseases based on color analysis"""
        issues = []
        confidences = []
        
        green_health = color_analysis['green_health_percentage']
        brown_yellow = color_analysis['brown_yellow_percentage']
        dark_spots = color_analysis['dark_spots_percentage']
        white_areas = color_analysis['white_areas_percentage']
        
        # Disease detection logic based on color patterns
        if brown_yellow > 20 and dark_spots > 10:
            issues.append({
                'type': 'disease',
                'name': 'Leaf Spot Disease',
                'description': 'Brown/yellow discoloration with dark spots detected',
                'severity': 'moderate' if brown_yellow < 40 else 'severe',
                'affected_area': f"{brown_yellow:.1f}% of visible area"
            })
            confidences.append(0.75)
        
        if white_areas > 15:
            issues.append({
                'type': 'disease',
                'name': 'Fungal Infection',
                'description': 'White fungal growth detected on plant surface',
                'severity': 'moderate' if white_areas < 30 else 'severe',
                'affected_area': f"{white_areas:.1f}% of visible area"
            })
            confidences.append(0.70)
        
        if green_health < 30:
            issues.append({
                'type': 'general_stress',
                'name': 'Plant Stress',
                'description': 'Low green vegetation indicates plant stress or disease',
                'severity': 'severe' if green_health < 15 else 'moderate',
                'affected_area': f"{100 - green_health:.1f}% affected"
            })
            confidences.append(0.65)
        
        # Crop-specific disease detection
        if crop_type and crop_type in self.disease_database:
            crop_diseases = self.disease_database[crop_type]
            
            if brown_yellow > 15 and crop_type == 'rice':
                issues.append({
                    'type': 'disease',
                    'name': 'Brown Spot',
                    'description': 'Symptoms consistent with rice brown spot disease',
                    'severity': 'moderate',
                    'crop_specific': True
                })
                confidences.append(0.68)
            
            if dark_spots > 8 and crop_type == 'tomato':
                issues.append({
                    'type': 'disease',
                    'name': 'Early Blight',
                    'description': 'Dark spots with possible concentric rings - early blight symptoms',
                    'severity': 'moderate',
                    'crop_specific': True
                })
                confidences.append(0.72)
        
        return {
            'issues': issues,
            'confidences': confidences
        }
    
    def _detect_pests_from_features(self, shape_analysis: Dict, texture_analysis: Dict, crop_type: str = None) -> Dict:
        """Detect pests based on shape and texture features"""
        issues = []
        confidences = []
        
        pest_count = shape_analysis['potential_pest_count']
        spot_count = texture_analysis['detected_spots']
        
        # Pest detection logic
        if pest_count > 5:
            issues.append({
                'type': 'pest',
                'name': 'Small Pest Infestation',
                'description': f'Multiple small objects detected ({pest_count} potential pests)',
                'severity': 'moderate' if pest_count < 15 else 'severe',
                'pest_count': pest_count
            })
            confidences.append(0.60)
        
        if spot_count > 20:
            issues.append({
                'type': 'damage',
                'name': 'Feeding Damage',
                'description': f'Multiple damage spots detected ({spot_count} spots)',
                'severity': 'moderate' if spot_count < 50 else 'severe',
                'damage_type': 'pest_feeding'
            })
            confidences.append(0.55)
        
        # Analyze pest shapes for specific identification
        for pest_feature in shape_analysis.get('pest_features', [])[:5]:
            circularity = pest_feature['circularity']
            aspect_ratio = pest_feature['aspect_ratio']
            area = pest_feature['area']
            
            # Aphid-like characteristics
            if 0.6 < circularity < 1.0 and 0.8 < aspect_ratio < 1.5 and 50 < area < 300:
                issues.append({
                    'type': 'pest',
                    'name': 'Aphid-like Pest',
                    'description': 'Small rounded pest detected - possibly aphids',
                    'severity': 'mild',
                    'characteristics': 'small, rounded shape'
                })
                confidences.append(0.50)
                break
            
            # Caterpillar-like characteristics
            elif aspect_ratio > 2.0 and area > 500:
                issues.append({
                    'type': 'pest',
                    'name': 'Caterpillar',
                    'description': 'Elongated pest detected - possibly caterpillar',
                    'severity': 'moderate',
                    'characteristics': 'elongated shape, larger size'
                })
                confidences.append(0.55)
                break
        
        return {
            'issues': issues,
            'confidences': confidences
        }
    
    def _general_plant_assessment(self, color_analysis: Dict, crop_type: str = None) -> Dict:
        """Provide general plant health assessment"""
        green_health = color_analysis['green_health_percentage']
        
        if green_health > 70:
            assessment = {
                'type': 'health_assessment',
                'name': 'Healthy Plant',
                'description': 'Plant appears healthy with good green coloration',
                'severity': 'none',
                'health_score': 'excellent'
            }
            confidence = 0.80
        elif green_health > 50:
            assessment = {
                'type': 'health_assessment',
                'name': 'Moderately Healthy',
                'description': 'Plant shows some stress but generally healthy',
                'severity': 'mild',
                'health_score': 'good'
            }
            confidence = 0.70
        else:
            assessment = {
                'type': 'health_assessment',
                'name': 'Plant Stress',
                'description': 'Plant shows signs of stress - monitor closely',
                'severity': 'moderate',
                'health_score': 'poor'
            }
            confidence = 0.75
        
        return {
            'assessment': assessment,
            'confidence': confidence
        }
    
    def _determine_severity(self, detected_issues: List[Dict]) -> str:
        """Determine overall severity based on detected issues"""
        if not detected_issues:
            return 'none'
        
        severities = [issue.get('severity', 'mild') for issue in detected_issues]
        
        if 'severe' in severities:
            return 'severe'
        elif 'moderate' in severities:
            return 'moderate'
        else:
            return 'mild'
    
    def _generate_treatment_recommendations(self, detected_issues: List[Dict], crop_type: str = None) -> List[str]:
        """Generate treatment recommendations based on detected issues"""
        recommendations = []
        
        for issue in detected_issues:
            issue_type = issue.get('type', 'unknown')
            issue_name = issue.get('name', 'Unknown Issue')
            severity = issue.get('severity', 'mild')
            
            if issue_type == 'disease':
                if 'fungal' in issue_name.lower() or 'blight' in issue_name.lower():
                    recommendations.extend([
                        "🍃 Apply neem oil spray (2ml/liter) in evening",
                        "🧪 Use copper-based fungicide if organic treatment fails",
                        "💨 Improve air circulation around plants",
                        "💧 Avoid overhead watering"
                    ])
                elif 'spot' in issue_name.lower():
                    recommendations.extend([
                        "🍃 Apply Trichoderma-based bio-fungicide",
                        "🧪 Use Mancozeb (2gm/liter) for severe infections",
                        "🌱 Remove affected leaves and destroy",
                        "🔄 Practice crop rotation next season"
                    ])
                elif 'bacterial' in issue_name.lower():
                    recommendations.extend([
                        "🧪 Apply copper oxychloride (3gm/liter)",
                        "💧 Avoid water splash on leaves",
                        "🌱 Remove infected plant parts",
                        "🧂 Apply balanced fertilization"
                    ])
            
            elif issue_type == 'pest':
                if 'aphid' in issue_name.lower():
                    recommendations.extend([
                        "🍃 Spray neem oil solution (3ml/liter)",
                        "🧪 Use Imidacloprid (0.5ml/liter) for severe infestation",
                        "🐞 Release ladybird beetles for biological control",
                        "💧 Use soap water spray (mild detergent)"
                    ])
                elif 'caterpillar' in issue_name.lower() or 'bollworm' in issue_name.lower():
                    recommendations.extend([
                        "🦠 Apply Bt spray (1-2gm/liter)",
                        "🧪 Use Chlorpyrifos (2ml/liter) if needed",
                        "🔍 Manual collection during early morning",
                        "🌱 Install pheromone traps"
                    ])
                elif 'whitefly' in issue_name.lower():
                    recommendations.extend([
                        "🍃 Neem oil spray (2ml/liter) regularly",
                        "💛 Install yellow sticky traps",
                        "🧪 Use Thiamethoxam (0.3gm/liter) if severe",
                        "🌱 Remove heavily infested leaves"
                    ])
            
            elif issue_type == 'general_stress':
                recommendations.extend([
                    "💧 Check soil moisture and irrigation schedule",
                    "🧪 Test soil for nutrient deficiencies",
                    "🌡️ Monitor temperature stress",
                    "🧂 Ensure balanced fertilization"
                ])
        
        # Add general preventive measures
        recommendations.extend([
            "📊 Monitor plants weekly for early detection",
            "🧼 Maintain field hygiene and sanitation",
            "🔄 Follow integrated pest management (IPM)",
            "📞 Consult local agriculture extension officer if needed"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Return top 10 recommendations
    
    def _save_detection_result(self, result: DetectionResult) -> None:
        """Save detection result for future reference"""
        try:
            result_data = {
                'detection_id': result.detection_id,
                'image_path': result.image_path,
                'detected_issues': result.detected_issues,
                'confidence_scores': result.confidence_scores,
                'treatment_recommendations': result.treatment_recommendations,
                'severity': result.severity,
                'crop_type': result.crop_type,
                'timestamp': result.timestamp.isoformat()
            }
            
            # Save to JSON file (in production, use database)
            results_file = 'static/detection_results.json'
            
            # Load existing results
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    existing_results = json.load(f)
            else:
                existing_results = []
            
            # Add new result
            existing_results.append(result_data)
            
            # Keep only last 100 results
            if len(existing_results) > 100:
                existing_results = existing_results[-100:]
            
            # Save updated results
            with open(results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving detection result: {str(e)}")
    
    def get_detection_history(self, limit: int = 20) -> List[Dict]:
        """Get detection history"""
        try:
            results_file = 'static/detection_results.json'
            
            if not os.path.exists(results_file):
                return []
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Return most recent results
            return results[-limit:]
            
        except Exception as e:
            logger.error(f"Error getting detection history: {str(e)}")
            return []
    
    def get_disease_info(self, disease_name: str, crop_type: str = None) -> Dict:
        """Get detailed information about a specific disease"""
        try:
            # Search in disease database
            if crop_type and crop_type in self.disease_database:
                crop_diseases = self.disease_database[crop_type]
                
                for disease_key, disease_info in crop_diseases.items():
                    if disease_name.lower() in disease_key.lower():
                        return {
                            'disease_name': disease_key.replace('_', ' ').title(),
                            'crop': crop_type,
                            'information': disease_info,
                            'treatment_options': self._get_disease_treatments(disease_key),
                            'prevention_measures': self._get_prevention_measures(disease_key)
                        }
            
            # Generic search across all crops
            for crop, diseases in self.disease_database.items():
                for disease_key, disease_info in diseases.items():
                    if disease_name.lower() in disease_key.lower():
                        return {
                            'disease_name': disease_key.replace('_', ' ').title(),
                            'crop': crop,
                            'information': disease_info,
                            'treatment_options': self._get_disease_treatments(disease_key),
                            'prevention_measures': self._get_prevention_measures(disease_key)
                        }
            
            return {
                'error': f'Disease "{disease_name}" not found in database',
                'suggestions': list(self._get_all_disease_names())
            }
            
        except Exception as e:
            logger.error(f"Error getting disease info: {str(e)}")
            return {'error': str(e)}
    
    def _get_disease_treatments(self, disease_key: str) -> Dict:
        """Get treatment options for a specific disease"""
        # Map diseases to treatment categories
        fungal_diseases = ['blast', 'brown_spot', 'rust', 'blight', 'spot']
        bacterial_diseases = ['bacterial_blight', 'bacterial_wilt']
        
        treatments = {
            'organic': [],
            'chemical': [],
            'cultural': []
        }
        
        if any(keyword in disease_key for keyword in fungal_diseases):
            treatments['organic'].extend(['neem_oil', 'trichoderma'])
            treatments['chemical'].extend(['mancozeb'])
            treatments['cultural'].extend(['crop_rotation', 'clean_cultivation'])
        
        if any(keyword in disease_key for keyword in bacterial_diseases):
            treatments['chemical'].extend(['copper_oxychloride'])
            treatments['cultural'].extend(['avoid_overhead_irrigation', 'remove_infected_parts'])
        
        # Get detailed treatment info
        detailed_treatments = {}
        for category, treatment_list in treatments.items():
            detailed_treatments[category] = []
            for treatment in treatment_list:
                if category == 'organic' and treatment in self.treatment_database['organic_treatments']:
                    detailed_treatments[category].append(self.treatment_database['organic_treatments'][treatment])
                elif category == 'chemical' and treatment in self.treatment_database['chemical_treatments']:
                    detailed_treatments[category].append(self.treatment_database['chemical_treatments'][treatment])
                elif category == 'cultural' and treatment in self.treatment_database['cultural_practices']:
                    detailed_treatments[category].append(self.treatment_database['cultural_practices'][treatment])
        
        return detailed_treatments
    
    def _get_prevention_measures(self, disease_key: str) -> List[str]:
        """Get prevention measures for a disease"""
        general_prevention = [
            "Use disease-resistant varieties when available",
            "Maintain proper plant spacing for air circulation",
            "Avoid overhead irrigation during humid conditions",
            "Remove and destroy infected plant debris",
            "Practice crop rotation with non-host crops",
            "Monitor plants regularly for early detection",
            "Maintain optimal soil nutrition and pH",
            "Use certified disease-free seeds/seedlings"
        ]
        
        # Add disease-specific prevention
        if 'fungal' in disease_key or any(keyword in disease_key for keyword in ['blast', 'spot', 'rust', 'blight']):
            general_prevention.extend([
                "Avoid working in fields when plants are wet",
                "Ensure good drainage to prevent waterlogging",
                "Apply preventive fungicide sprays during favorable disease conditions"
            ])
        
        if 'bacterial' in disease_key:
            general_prevention.extend([
                "Disinfect tools between plants",
                "Avoid creating wounds on plants",
                "Control insect vectors that spread bacteria"
            ])
        
        return general_prevention[:8]  # Return top 8 prevention measures
    
    def _get_all_disease_names(self) -> List[str]:
        """Get all disease names from database"""
        diseases = []
        for crop, crop_diseases in self.disease_database.items():
            for disease_key in crop_diseases.keys():
                disease_name = disease_key.replace('_', ' ').title()
                diseases.append(f"{disease_name} ({crop.title()})")
        return diseases

# Utility functions for image processing
def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        return ""

def decode_base64_to_image(base64_string: str) -> bytes:
    """Decode base64 string to image bytes"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        return base64.b64decode(base64_string)
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise

def resize_image_for_analysis(image_path: str, max_size: Tuple[int, int] = (800, 600)) -> str:
    """Resize image for optimal analysis"""
    try:
        with Image.open(image_path) as img:
            # Calculate new size maintaining aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save resized image
            resized_path = image_path.replace('.jpg', '_resized.jpg')
            img.save(resized_path, 'JPEG', quality=85)
            
            return resized_path
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image_path

# Plant health monitoring functions
def analyze_plant_health_trends(detection_results: List[Dict], crop_type: str = None) -> Dict:
    """Analyze plant health trends from multiple detections"""
    try:
        if not detection_results:
            return {'error': 'No detection data available'}
        
        # Filter by crop type if specified
        if crop_type:
            detection_results = [r for r in detection_results if r.get('crop_type') == crop_type]
        
        if not detection_results:
            return {'error': f'No detection data for crop type: {crop_type}'}
        
        # Analyze trends
        total_detections = len(detection_results)
        disease_detections = len([r for r in detection_results if any(issue['type'] == 'disease' for issue in r['detected_issues'])])
        pest_detections = len([r for r in detection_results if any(issue['type'] == 'pest' for issue in r['detected_issues'])])
        healthy_detections = len([r for r in detection_results if r['severity'] == 'none'])
        
        # Calculate severity distribution
        severity_counts = {'none': 0, 'mild': 0, 'moderate': 0, 'severe': 0}
        for result in detection_results:
            severity = result.get('severity', 'none')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Recent trend (last 5 detections)
        recent_results = detection_results[-5:]
        recent_severity_trend = [r.get('severity', 'none') for r in recent_results]
        
        trend_analysis = {
            'total_detections': total_detections,
            'health_statistics': {
                'disease_rate': round((disease_detections / total_detections) * 100, 1),
                'pest_rate': round((pest_detections / total_detections) * 100, 1),
                'healthy_rate': round((healthy_detections / total_detections) * 100, 1)
            },
            'severity_distribution': severity_counts,
            'recent_trend': recent_severity_trend,
            'trend_direction': _calculate_trend_direction(recent_severity_trend),
            'recommendations': _generate_trend_recommendations(severity_counts, recent_severity_trend)
        }
        
        return trend_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing plant health trends: {str(e)}")
        return {'error': str(e)}

def _calculate_trend_direction(recent_trend: List[str]) -> str:
    """Calculate health trend direction"""
    severity_values = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    
    if len(recent_trend) < 3:
        return 'insufficient_data'
    
    trend_values = [severity_values[severity] for severity in recent_trend]
    
    # Simple trend calculation
    early_avg = sum(trend_values[:2]) / 2
    late_avg = sum(trend_values[-2:]) / 2
    
    if late_avg > early_avg + 0.5:
        return 'deteriorating'
    elif late_avg < early_avg - 0.5:
        return 'improving'
    else:
        return 'stable'

def _generate_trend_recommendations(severity_counts: Dict, recent_trend: List[str]) -> List[str]:
    """Generate recommendations based on health trends"""
    recommendations = []
    
    total = sum(severity_counts.values())
    severe_rate = (severity_counts.get('severe', 0) / total) * 100 if total > 0 else 0
    
    if severe_rate > 30:
        recommendations.extend([
            "🚨 High disease/pest pressure detected - implement intensive monitoring",
            "🧪 Consider preventive treatment applications",
            "👨‍🌾 Consult agricultural extension officer immediately"
        ])
    elif severe_rate > 15:
        recommendations.extend([
            "⚠️ Moderate disease pressure - increase monitoring frequency",
            "🍃 Focus on organic prevention methods"
        ])
    
    if 'severe' in recent_trend[-2:]:
        recommendations.append("📈 Recent severe issues detected - take immediate action")
    
    if recent_trend.count('none') >= 3:
        recommendations.append("✅ Good plant health trend - continue current practices")
    
    return recommendations

# Global detector instance
plant_detector = PlantDiseaseDetector()

def detect_disease_from_image(image_data: bytes, crop_type: str = None) -> Dict:
    """Main function to detect diseases from uploaded image"""
    try:
        result = plant_detector.analyze_plant_image(image_data, crop_type)
        
        return {
            'detection_id': result.detection_id,
            'crop_type': result.crop_type,
            'detected_issues': result.detected_issues,
            'confidence_scores': result.confidence_scores,
            'treatment_recommendations': result.treatment_recommendations,
            'severity': result.severity,
            'timestamp': result.timestamp.isoformat(),
            'analysis_summary': _generate_analysis_summary(result),
            'next_steps': _get_next_steps_recommendations(result.severity, result.detected_issues)
        }
        
    except Exception as e:
        logger.error(f"Error in disease detection: {str(e)}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def _generate_analysis_summary(result: DetectionResult) -> str:
    """Generate a human-readable analysis summary"""
    issue_count = len(result.detected_issues)
    
    if issue_count == 0:
        return "No specific issues detected. Plant appears healthy."
    
    issue_types = set(issue['type'] for issue in result.detected_issues)
    
    if result.severity == 'severe':
        summary = f"⚠️ Severe issues detected: {issue_count} problems identified. "
    elif result.severity == 'moderate':
        summary = f"⚡ Moderate issues found: {issue_count} problems need attention. "
    else:
        summary = f"✓ Minor issues detected: {issue_count} minor problems found. "
    
    if 'disease' in issue_types and 'pest' in issue_types:
        summary += "Both disease and pest issues are present."
    elif 'disease' in issue_types:
        summary += "Disease-related issues detected."
    elif 'pest' in issue_types:
        summary += "Pest-related issues found."
    else:
        summary += "General plant health assessment completed."
    
    return summary

def _get_next_steps_recommendations(severity: str, detected_issues: List[Dict]) -> List[str]:
    """Get next steps based on detection results"""
    steps = []
    
    if severity == 'severe':
        steps.extend([
            "🚨 Take immediate action - implement recommended treatments",
            "📞 Contact local agricultural extension officer",
            "🔍 Monitor other plants in the field for similar symptoms"
        ])
    elif severity == 'moderate':
        steps.extend([
            "⚠️ Plan treatment within 2-3 days",
            "🔍 Inspect nearby plants for spread",
            "📊 Document symptoms for tracking"
        ])
    else:
        steps.extend([
            "📊 Continue regular monitoring",
            "🍃 Consider preventive organic treatments",
            "📅 Schedule follow-up inspection in 1 week"
        ])
    
    # Add issue-specific steps
    issue_types = set(issue['type'] for issue in detected_issues)
    
    if 'disease' in issue_types:
        steps.append("🧪 Apply appropriate fungicide or bactericide")
    
    if 'pest' in issue_types:
        steps.append("🐛 Implement pest control measures")
    
    steps.append("📸 Take follow-up photos after treatment to track progress")
    
    return steps[:6]  # Return top 6 next steps

# Initialize plant detector
plant_detector = PlantDiseaseDetector()
