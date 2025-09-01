#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Soil Health Assessment Module for Smart Farmer Application
Provides comprehensive soil analysis, pH recommendations, and fertilizer guidance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import requests
import json

logger = logging.getLogger(__name__)

class SoilHealthAnalyzer:
    """Advanced soil health analysis and recommendation system"""
    
    def __init__(self):
        self.soil_nutrient_standards = self._load_soil_standards()
        self.fertilizer_database = self._load_fertilizer_database()
        self.crop_nutrient_requirements = self._load_crop_requirements()
    
    def _load_soil_standards(self) -> Dict:
        """Load soil health standards and optimal ranges"""
        return {
            'ph_ranges': {
                'very_acidic': (0, 4.5),
                'acidic': (4.5, 6.0),
                'slightly_acidic': (6.0, 6.5),
                'neutral': (6.5, 7.5),
                'slightly_alkaline': (7.5, 8.5),
                'alkaline': (8.5, 9.5),
                'very_alkaline': (9.5, 14)
            },
            'nutrient_levels': {
                'nitrogen': {
                    'low': (0, 280),
                    'medium': (280, 560),
                    'high': (560, 1000)
                },
                'phosphorus': {
                    'low': (0, 11),
                    'medium': (11, 25),
                    'high': (25, 50)
                },
                'potassium': {
                    'low': (0, 110),
                    'medium': (110, 280),
                    'high': (280, 500)
                }
            },
            'organic_carbon': {
                'low': (0, 0.5),
                'medium': (0.5, 0.75),
                'high': (0.75, 2.0)
            },
            'electrical_conductivity': {
                'normal': (0, 2.0),
                'slightly_saline': (2.0, 4.0),
                'moderately_saline': (4.0, 8.0),
                'highly_saline': (8.0, 16.0)
            }
        }
    
    def _load_fertilizer_database(self) -> Dict:
        """Load fertilizer types and their nutrient content"""
        return {
            'organic': {
                'farmyard_manure': {'N': 0.5, 'P': 0.2, 'K': 0.5, 'organic_matter': 20},
                'compost': {'N': 1.5, 'P': 1.0, 'K': 1.5, 'organic_matter': 30},
                'vermicompost': {'N': 1.8, 'P': 1.3, 'K': 1.8, 'organic_matter': 35},
                'green_manure': {'N': 2.0, 'P': 0.5, 'K': 2.0, 'organic_matter': 25},
                'neem_cake': {'N': 5.0, 'P': 1.0, 'K': 1.4, 'organic_matter': 80},
                'bone_meal': {'N': 4.0, 'P': 20.0, 'K': 0.2, 'organic_matter': 60}
            },
            'chemical': {
                'urea': {'N': 46, 'P': 0, 'K': 0},
                'dap': {'N': 18, 'P': 46, 'K': 0},
                'mop': {'N': 0, 'P': 0, 'K': 60},
                'npk_complex': {'N': 17, 'P': 17, 'K': 17},
                'ssp': {'N': 0, 'P': 16, 'K': 0},
                'tsp': {'N': 0, 'P': 46, 'K': 0},
                'potash': {'N': 0, 'P': 0, 'K': 50}
            },
            'micronutrients': {
                'zinc_sulphate': {'Zn': 36, 'S': 18},
                'iron_sulphate': {'Fe': 20, 'S': 11},
                'borax': {'B': 11},
                'manganese_sulphate': {'Mn': 32, 'S': 19},
                'copper_sulphate': {'Cu': 25, 'S': 13}
            }
        }
    
    def _load_crop_requirements(self) -> Dict:
        """Load crop-specific nutrient requirements"""
        return {
            'rice': {
                'N': {'requirement': 120, 'timing': 'split_application'},
                'P': {'requirement': 60, 'timing': 'basal'},
                'K': {'requirement': 40, 'timing': 'split_application'},
                'optimal_ph': (5.5, 6.5),
                'organic_matter': 1.5
            },
            'wheat': {
                'N': {'requirement': 120, 'timing': 'split_application'},
                'P': {'requirement': 60, 'timing': 'basal'},
                'K': {'requirement': 40, 'timing': 'basal'},
                'optimal_ph': (6.0, 7.5),
                'organic_matter': 1.0
            },
            'cotton': {
                'N': {'requirement': 150, 'timing': 'split_application'},
                'P': {'requirement': 75, 'timing': 'basal'},
                'K': {'requirement': 75, 'timing': 'split_application'},
                'optimal_ph': (5.8, 8.0),
                'organic_matter': 1.2
            },
            'maize': {
                'N': {'requirement': 120, 'timing': 'split_application'},
                'P': {'requirement': 60, 'timing': 'basal'},
                'K': {'requirement': 40, 'timing': 'split_application'},
                'optimal_ph': (6.0, 7.5),
                'organic_matter': 1.2
            },
            'sugarcane': {
                'N': {'requirement': 200, 'timing': 'split_application'},
                'P': {'requirement': 80, 'timing': 'basal'},
                'K': {'requirement': 120, 'timing': 'split_application'},
                'optimal_ph': (6.5, 7.5),
                'organic_matter': 1.5
            },
            'soybean': {
                'N': {'requirement': 30, 'timing': 'basal'},  # Less N due to nitrogen fixation
                'P': {'requirement': 80, 'timing': 'basal'},
                'K': {'requirement': 40, 'timing': 'basal'},
                'optimal_ph': (6.0, 7.0),
                'organic_matter': 1.0
            },
            'chickpea': {
                'N': {'requirement': 25, 'timing': 'basal'},  # Nitrogen-fixing legume
                'P': {'requirement': 50, 'timing': 'basal'},
                'K': {'requirement': 30, 'timing': 'basal'},
                'optimal_ph': (6.0, 7.5),
                'organic_matter': 1.0
            }
        }
    
    def analyze_soil_health(self, soil_data: Dict) -> Dict:
        """Comprehensive soil health analysis"""
        try:
            analysis = {
                'overall_score': 0,
                'individual_scores': {},
                'recommendations': [],
                'deficiencies': [],
                'strengths': [],
                'fertilizer_plan': {},
                'amendment_suggestions': []
            }
            
            # Analyze pH
            ph_analysis = self._analyze_ph(soil_data.get('ph', 7.0))
            analysis['individual_scores']['ph'] = ph_analysis
            
            # Analyze major nutrients
            n_analysis = self._analyze_nutrient('nitrogen', soil_data.get('nitrogen', 300))
            p_analysis = self._analyze_nutrient('phosphorus', soil_data.get('phosphorus', 15))
            k_analysis = self._analyze_nutrient('potassium', soil_data.get('potassium', 200))
            
            analysis['individual_scores']['nitrogen'] = n_analysis
            analysis['individual_scores']['phosphorus'] = p_analysis
            analysis['individual_scores']['potassium'] = k_analysis
            
            # Analyze organic carbon
            oc_analysis = self._analyze_organic_carbon(soil_data.get('organic_carbon', 0.6))
            analysis['individual_scores']['organic_carbon'] = oc_analysis
            
            # Analyze salinity
            ec_analysis = self._analyze_electrical_conductivity(soil_data.get('electrical_conductivity', 1.0))
            analysis['individual_scores']['electrical_conductivity'] = ec_analysis
            
            # Calculate overall health score
            scores = [
                ph_analysis['score'],
                n_analysis['score'],
                p_analysis['score'],
                k_analysis['score'],
                oc_analysis['score'],
                ec_analysis['score']
            ]
            analysis['overall_score'] = round(sum(scores) / len(scores), 1)
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_soil_recommendations(analysis['individual_scores'])
            
            # Identify deficiencies and strengths
            for nutrient, data in analysis['individual_scores'].items():
                if data['level'] == 'low':
                    analysis['deficiencies'].append({
                        'nutrient': nutrient,
                        'severity': data['score'],
                        'action': data['recommendation']
                    })
                elif data['level'] == 'high':
                    analysis['strengths'].append({
                        'nutrient': nutrient,
                        'score': data['score']
                    })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in soil health analysis: {str(e)}")
            raise
    
    def _analyze_ph(self, ph_value: float) -> Dict:
        """Analyze soil pH and provide recommendations"""
        ranges = self.soil_nutrient_standards['ph_ranges']
        
        for level, (min_val, max_val) in ranges.items():
            if min_val <= ph_value < max_val:
                if level in ['neutral', 'slightly_acidic']:
                    score = 10
                    recommendation = "pH level is optimal for most crops"
                elif level in ['acidic', 'slightly_alkaline']:
                    score = 7
                    recommendation = f"pH is {level.replace('_', ' ')}. Consider soil amendments"
                else:
                    score = 4
                    recommendation = f"pH is {level.replace('_', ' ')}. Urgent soil treatment needed"
                
                return {
                    'value': ph_value,
                    'level': level,
                    'score': score,
                    'recommendation': recommendation,
                    'amendment': self._get_ph_amendment(ph_value)
                }
        
        return {
            'value': ph_value,
            'level': 'unknown',
            'score': 5,
            'recommendation': 'Please verify pH measurement',
            'amendment': 'Retest soil pH'
        }
    
    def _analyze_nutrient(self, nutrient: str, value: float) -> Dict:
        """Analyze individual nutrient levels"""
        ranges = self.soil_nutrient_standards['nutrient_levels'][nutrient]
        
        for level, (min_val, max_val) in ranges.items():
            if min_val <= value < max_val:
                if level == 'medium':
                    score = 10
                    recommendation = f"{nutrient.title()} level is adequate"
                elif level == 'high':
                    score = 8
                    recommendation = f"{nutrient.title()} level is high - reduce fertilizer input"
                else:  # low
                    score = 5
                    recommendation = f"{nutrient.title()} is deficient - increase fertilizer application"
                
                return {
                    'value': value,
                    'level': level,
                    'score': score,
                    'recommendation': recommendation
                }
        
        return {
            'value': value,
            'level': 'very_high',
            'score': 6,
            'recommendation': f"{nutrient.title()} level is very high - avoid over-fertilization"
        }
    
    def _analyze_organic_carbon(self, oc_value: float) -> Dict:
        """Analyze organic carbon content"""
        ranges = self.soil_nutrient_standards['organic_carbon']
        
        for level, (min_val, max_val) in ranges.items():
            if min_val <= oc_value < max_val:
                if level == 'high':
                    score = 10
                    recommendation = "Excellent organic matter content"
                elif level == 'medium':
                    score = 7
                    recommendation = "Good organic matter, consider adding more compost"
                else:  # low
                    score = 4
                    recommendation = "Low organic matter - urgent need for organic amendments"
                
                return {
                    'value': oc_value,
                    'level': level,
                    'score': score,
                    'recommendation': recommendation
                }
        
        return {
            'value': oc_value,
            'level': 'very_high',
            'score': 10,
            'recommendation': "Excellent organic matter content"
        }
    
    def _analyze_electrical_conductivity(self, ec_value: float) -> Dict:
        """Analyze soil salinity through electrical conductivity"""
        ranges = self.soil_nutrient_standards['electrical_conductivity']
        
        for level, (min_val, max_val) in ranges.items():
            if min_val <= ec_value < max_val:
                if level == 'normal':
                    score = 10
                    recommendation = "Soil salinity is normal"
                elif level == 'slightly_saline':
                    score = 7
                    recommendation = "Slightly saline soil - monitor and improve drainage"
                elif level == 'moderately_saline':
                    score = 5
                    recommendation = "Moderately saline - use salt-tolerant crops and improve drainage"
                else:  # highly_saline
                    score = 3
                    recommendation = "Highly saline soil - extensive reclamation needed"
                
                return {
                    'value': ec_value,
                    'level': level,
                    'score': score,
                    'recommendation': recommendation
                }
        
        return {
            'value': ec_value,
            'level': 'extremely_saline',
            'score': 2,
            'recommendation': "Extremely saline soil - professional soil reclamation required"
        }
    
    def _get_ph_amendment(self, ph_value: float) -> str:
        """Get pH amendment recommendations"""
        if ph_value < 6.0:
            return "Apply agricultural lime (CaCO3) at 2-4 quintals per hectare"
        elif ph_value > 8.0:
            return "Apply gypsum (CaSO4) at 2-3 quintals per hectare or organic matter"
        else:
            return "No pH amendment needed"
    
    def _generate_soil_recommendations(self, scores: Dict) -> List[str]:
        """Generate comprehensive soil improvement recommendations"""
        recommendations = []
        
        # pH recommendations
        ph_data = scores['ph']
        if ph_data['score'] < 8:
            recommendations.append(f"🧪 pH Management: {ph_data['amendment']}")
        
        # Nutrient recommendations
        for nutrient in ['nitrogen', 'phosphorus', 'potassium']:
            data = scores[nutrient]
            if data['level'] == 'low':
                recommendations.append(f"🌱 {nutrient.title()}: {data['recommendation']}")
        
        # Organic matter recommendations
        oc_data = scores['organic_carbon']
        if oc_data['score'] < 7:
            recommendations.append("🍃 Organic Matter: Add 5-10 tons of compost or vermicompost per hectare")
        
        # Salinity recommendations
        ec_data = scores['electrical_conductivity']
        if ec_data['score'] < 8:
            recommendations.append(f"🧂 Salinity: {ec_data['recommendation']}")
        
        # General recommendations
        recommendations.extend([
            "💧 Water Management: Ensure proper drainage and irrigation scheduling",
            "🔄 Crop Rotation: Practice crop rotation to maintain soil health",
            "🌾 Cover Crops: Use cover crops during fallow periods",
            "📊 Regular Testing: Test soil every 2-3 years for monitoring"
        ])
        
        return recommendations
    
    def calculate_fertilizer_requirement(self, soil_data: Dict, crop: str, area: float) -> Dict:
        """Calculate precise fertilizer requirements for a crop"""
        try:
            if crop not in self.crop_nutrient_requirements:
                crop = 'rice'  # Default fallback
            
            crop_req = self.crop_nutrient_requirements[crop]
            
            # Current soil nutrient levels
            soil_n = soil_data.get('nitrogen', 300)
            soil_p = soil_data.get('phosphorus', 15)
            soil_k = soil_data.get('potassium', 200)
            
            # Calculate nutrient gaps
            n_gap = max(0, crop_req['N']['requirement'] - (soil_n * 0.1))  # Convert mg/kg to kg/ha
            p_gap = max(0, crop_req['P']['requirement'] - (soil_p * 2))   # Convert mg/kg to kg/ha
            k_gap = max(0, crop_req['K']['requirement'] - (soil_k * 0.08)) # Convert mg/kg to kg/ha
            
            # Calculate fertilizer quantities
            fertilizer_plan = {}
            
            # Organic recommendations (30% of requirement)
            organic_n = n_gap * 0.3
            organic_p = p_gap * 0.3
            organic_k = k_gap * 0.3
            
            compost_needed = max(organic_n / 1.5, organic_p / 1.0, organic_k / 1.5) * 100  # kg
            fertilizer_plan['organic'] = {
                'compost': round(compost_needed * area, 0),
                'timing': '2-3 weeks before planting'
            }
            
            # Chemical fertilizer requirements (70% of remaining)
            remaining_n = n_gap * 0.7
            remaining_p = p_gap * 0.7
            remaining_k = k_gap * 0.7
            
            # Calculate chemical fertilizers
            dap_needed = min(remaining_p / 0.46, remaining_n / 0.18) * area  # kg
            urea_needed = max(0, (remaining_n - (dap_needed * 0.18)) / 0.46) * area  # kg
            mop_needed = remaining_k / 0.60 * area  # kg
            
            fertilizer_plan['chemical'] = {
                'dap': round(dap_needed, 1),
                'urea': round(urea_needed, 1),
                'mop': round(mop_needed, 1),
                'application_schedule': self._get_application_schedule(crop)
            }
            
            # Calculate costs (approximate Indian market rates)
            costs = {
                'compost': fertilizer_plan['organic']['compost'] * 8,  # Rs per kg
                'dap': fertilizer_plan['chemical']['dap'] * 28,
                'urea': fertilizer_plan['chemical']['urea'] * 6,
                'mop': fertilizer_plan['chemical']['mop'] * 18
            }
            
            total_cost = sum(costs.values())
            
            return {
                'crop': crop,
                'area': area,
                'nutrient_gaps': {
                    'nitrogen': round(n_gap, 1),
                    'phosphorus': round(p_gap, 1),
                    'potassium': round(k_gap, 1)
                },
                'fertilizer_plan': fertilizer_plan,
                'cost_analysis': {
                    'individual_costs': costs,
                    'total_cost': round(total_cost, 2),
                    'cost_per_hectare': round(total_cost / area, 2)
                },
                'application_timeline': self._generate_application_timeline(crop),
                'micronutrient_recommendations': self._get_micronutrient_recommendations(soil_data, crop)
            }
            
        except Exception as e:
            logger.error(f"Error calculating fertilizer requirement: {str(e)}")
            raise
    
    def _get_application_schedule(self, crop: str) -> Dict:
        """Get fertilizer application schedule for crop"""
        schedules = {
            'rice': {
                'basal': '100% P, 50% K, 25% N',
                'tillering': '50% N',
                'panicle_initiation': '25% N, 50% K'
            },
            'wheat': {
                'basal': '100% P, 100% K, 50% N',
                'crown_root_initiation': '25% N',
                'jointing': '25% N'
            },
            'cotton': {
                'basal': '100% P, 25% K, 25% N',
                'squaring': '50% N, 50% K',
                'flowering': '25% N, 25% K'
            },
            'maize': {
                'basal': '100% P, 50% K, 25% N',
                'knee_high': '50% N',
                'tasseling': '25% N, 50% K'
            }
        }
        
        return schedules.get(crop, schedules['rice'])
    
    def _generate_application_timeline(self, crop: str) -> List[Dict]:
        """Generate detailed fertilizer application timeline"""
        base_date = datetime.now()
        
        timelines = {
            'rice': [
                {'stage': 'Pre-planting', 'days': -15, 'activity': 'Apply compost and prepare soil'},
                {'stage': 'Transplanting', 'days': 0, 'activity': 'Apply basal fertilizers'},
                {'stage': 'Tillering', 'days': 25, 'activity': 'First top dressing of urea'},
                {'stage': 'Panicle initiation', 'days': 50, 'activity': 'Second top dressing'}
            ],
            'wheat': [
                {'stage': 'Pre-sowing', 'days': -10, 'activity': 'Apply organic manure'},
                {'stage': 'Sowing', 'days': 0, 'activity': 'Apply basal fertilizers'},
                {'stage': 'Crown root stage', 'days': 21, 'activity': 'First urea application'},
                {'stage': 'Jointing stage', 'days': 45, 'activity': 'Second urea application'}
            ],
            'cotton': [
                {'stage': 'Pre-planting', 'days': -20, 'activity': 'Prepare soil with organic matter'},
                {'stage': 'Sowing', 'days': 0, 'activity': 'Apply basal dose'},
                {'stage': 'Squaring', 'days': 45, 'activity': 'First top dressing'},
                {'stage': 'Flowering', 'days': 75, 'activity': 'Second top dressing'}
            ]
        }
        
        timeline = timelines.get(crop, timelines['rice'])
        
        for entry in timeline:
            target_date = base_date + timedelta(days=entry['days'])
            entry['date'] = target_date.strftime("%Y-%m-%d")
            entry['month'] = target_date.strftime("%B")
        
        return timeline
    
    def _get_micronutrient_recommendations(self, soil_data: Dict, crop: str) -> Dict:
        """Recommend micronutrients based on soil and crop"""
        recommendations = {}
        
        # Common micronutrient deficiencies in Indian soils
        if soil_data.get('zinc', 0.8) < 1.0:
            recommendations['zinc'] = {
                'deficiency': 'likely',
                'fertilizer': 'Zinc Sulphate',
                'quantity': '25 kg/hectare',
                'application': 'Mix with compost and apply basally'
            }
        
        if soil_data.get('iron', 4.0) < 4.5:
            recommendations['iron'] = {
                'deficiency': 'possible',
                'fertilizer': 'Iron Sulphate',
                'quantity': '20 kg/hectare',
                'application': 'Foliar spray during vegetative growth'
            }
        
        # Crop-specific micronutrients
        crop_specific = {
            'cotton': ['boron', 'zinc'],
            'rice': ['zinc', 'iron'],
            'wheat': ['zinc', 'iron'],
            'sugarcane': ['iron', 'manganese']
        }
        
        if crop in crop_specific:
            for micronutrient in crop_specific[crop]:
                if micronutrient not in recommendations:
                    recommendations[micronutrient] = {
                        'importance': f'Important for {crop}',
                        'preventive_dose': f'{micronutrient.title()} Sulphate - 10 kg/hectare'
                    }
        
        return recommendations
    
    def generate_soil_health_report(self, soil_data: Dict, crop: str = None, area: float = 1.0) -> Dict:
        """Generate comprehensive soil health report"""
        try:
            # Basic soil analysis
            analysis = self.analyze_soil_health(soil_data)
            
            # Fertilizer recommendations if crop is specified
            fertilizer_plan = {}
            if crop:
                fertilizer_plan = self.calculate_fertilizer_requirement(soil_data, crop, area)
            
            # Generate improvement plan
            improvement_plan = self._generate_improvement_plan(analysis, crop)
            
            # Cost-benefit analysis
            economic_analysis = self._calculate_economic_impact(analysis, fertilizer_plan, area)
            
            report = {
                'report_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'soil_analysis': analysis,
                'fertilizer_plan': fertilizer_plan,
                'improvement_plan': improvement_plan,
                'economic_analysis': economic_analysis,
                'next_steps': self._get_next_steps(analysis),
                'testing_recommendations': self._get_testing_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating soil health report: {str(e)}")
            raise

    def _generate_improvement_plan(self, analysis: Dict, crop: str = None) -> Dict:
        """Generate step-by-step soil improvement plan"""
        plan = {
            'immediate_actions': [],
            'short_term': [],  # 1-3 months
            'long_term': []    # 6-12 months
        }
        
        overall_score = analysis['overall_score']
        
        if overall_score < 6:
            plan['immediate_actions'].extend([
                "Stop all fertilizer applications until soil test results are confirmed",
                "Implement drainage improvement if waterlogging is observed",
                "Start composting organic waste for soil amendment"
            ])
        
        # pH corrections
        ph_score = analysis['individual_scores']['ph']['score']
        if ph_score < 7:
            plan['immediate_actions'].append(f"Apply lime/gypsum: {analysis['individual_scores']['ph']['amendment']}")
        
        # Nutrient deficiencies
        for deficiency in analysis['deficiencies']:
            if deficiency['severity'] < 6:
                plan['immediate_actions'].append(f"Address {deficiency['nutrient']} deficiency: {deficiency['action']}")
        
        # Short-term improvements
        plan['short_term'].extend([
            "Apply organic matter (compost/vermicompost) regularly",
            "Implement proper irrigation scheduling",
            "Monitor crop response to amendments"
        ])
        
        # Long-term sustainability
        plan['long_term'].extend([
            "Establish crop rotation cycle",
            "Build long-term organic matter content",
            "Monitor soil health trends annually"
        ])
        
        return plan
    
    def _calculate_economic_impact(self, analysis: Dict, fertilizer_plan: Dict, area: float) -> Dict:
        """Calculate economic impact of soil health improvements"""
        try:
            current_productivity_factor = analysis['overall_score'] / 10
            
            # Potential yield improvement
            if analysis['overall_score'] < 7:
                potential_improvement = (8 - analysis['overall_score']) * 10  # % improvement
            else:
                potential_improvement = 5  # Marginal improvement for good soils
            
            # Cost calculations
            improvement_cost = 0
            if fertilizer_plan:
                improvement_cost = fertilizer_plan.get('cost_analysis', {}).get('total_cost', 0)
            
            # Add soil amendment costs
            for deficiency in analysis['deficiencies']:
                if deficiency['severity'] < 6:
                    improvement_cost += 2000 * area  # Approximate amendment cost per hectare
            
            # Revenue impact (based on average crop value)
            avg_crop_value_per_hectare = 50000  # Rs per hectare (conservative estimate)
            potential_additional_revenue = (potential_improvement / 100) * avg_crop_value_per_hectare * area
            
            roi = (potential_additional_revenue - improvement_cost) / improvement_cost * 100 if improvement_cost > 0 else 0
            
            return {
                'current_productivity_factor': round(current_productivity_factor, 2),
                'potential_yield_improvement': round(potential_improvement, 1),
                'improvement_cost': round(improvement_cost, 2),
                'potential_additional_revenue': round(potential_additional_revenue, 2),
                'roi_percentage': round(roi, 1),
                'payback_period_months': round(improvement_cost / (potential_additional_revenue / 12), 1) if potential_additional_revenue > 0 else 0,
                'profitability': 'High' if roi > 50 else 'Medium' if roi > 20 else 'Low'
            }
            
        except Exception as e:
            logger.error(f"Error in economic analysis: {str(e)}")
            return {}
    
    def _get_next_steps(self, analysis: Dict) -> List[str]:
        """Get prioritized next steps based on soil analysis"""
        steps = []
        
        if analysis['overall_score'] < 5:
            steps.append("🚨 Priority: Address soil health urgently before next planting")
        elif analysis['overall_score'] < 7:
            steps.append("⚠️ Important: Implement soil improvement plan gradually")
        else:
            steps.append("✅ Maintain: Continue current soil management practices")
        
        # Add specific action items
        if analysis['deficiencies']:
            steps.append("🎯 Focus on addressing identified nutrient deficiencies")
        
        steps.extend([
            "📅 Schedule: Plan fertilizer applications according to crop calendar",
            "📊 Monitor: Track soil improvement through regular testing",
            "🌱 Optimize: Adjust practices based on crop response"
        ])
        
        return steps
    
    def _get_testing_recommendations(self) -> Dict:
        """Recommend soil testing frequency and parameters"""
        return {
            'frequency': {
                'basic_test': 'Every 2-3 years',
                'nutrient_test': 'Annually before crop season',
                'problem_diagnosis': 'When crop issues arise'
            },
            'essential_parameters': [
                'pH', 'Electrical Conductivity', 'Organic Carbon',
                'Available Nitrogen', 'Available Phosphorus', 'Available Potassium'
            ],
            'additional_parameters': [
                'Zinc', 'Iron', 'Manganese', 'Copper', 'Boron',
                'Sulphur', 'Calcium', 'Magnesium'
            ],
            'seasonal_focus': {
                'pre_kharif': 'Complete nutrient analysis',
                'pre_rabi': 'pH and major nutrients',
                'post_harvest': 'Organic matter and pH'
            }
        }

# Initialize soil health analyzer
soil_health_analyzer = SoilHealthAnalyzer()

def get_soil_data_from_government_api(state: str, district: str) -> Dict:
    """Fetch soil data from government APIs (placeholder for actual implementation)"""
    try:
        # This would integrate with actual government soil health APIs
        # For now, returning sample data structure
        
        # Simulate API call delay
        import time
        time.sleep(0.1)
        
        # Sample soil data (in real implementation, this would come from API)
        sample_data = {
            'ph': np.random.normal(6.8, 0.8),
            'nitrogen': np.random.normal(350, 100),
            'phosphorus': np.random.normal(18, 8),
            'potassium': np.random.normal(220, 80),
            'organic_carbon': np.random.normal(0.7, 0.3),
            'electrical_conductivity': np.random.normal(1.2, 0.5),
            'zinc': np.random.normal(1.1, 0.4),
            'iron': np.random.normal(4.8, 1.2),
            'data_source': 'Soil Health Card Database',
            'collection_date': datetime.now().strftime('%Y-%m-%d'),
            'reliability': 'high'
        }
        
        # Ensure values are within realistic ranges
        sample_data['ph'] = max(4.0, min(9.0, sample_data['ph']))
        sample_data['organic_carbon'] = max(0.1, min(2.0, sample_data['organic_carbon']))
        sample_data['electrical_conductivity'] = max(0.1, min(4.0, sample_data['electrical_conductivity']))
        
        logger.info(f"Retrieved soil data for {district}, {state}")
        return sample_data
        
    except Exception as e:
        logger.error(f"Error fetching government soil data: {str(e)}")
        # Return default soil data
        return {
            'ph': 6.8,
            'nitrogen': 300,
            'phosphorus': 15,
            'potassium': 200,
            'organic_carbon': 0.6,
            'electrical_conductivity': 1.0,
            'data_source': 'Default values',
            'collection_date': datetime.now().strftime('%Y-%m-%d'),
            'reliability': 'estimated'
        }

def get_soil_improvement_recommendations(current_soil: Dict, target_crop: str) -> Dict:
    """Get comprehensive soil improvement recommendations"""
    try:
        analyzer = SoilHealthAnalyzer()
        
        # Generate comprehensive analysis
        analysis = analyzer.analyze_soil_health(current_soil)
        
        # Calculate fertilizer requirements for target crop
        fertilizer_plan = analyzer.calculate_fertilizer_requirement(current_soil, target_crop, 1.0)
        
        # Generate improvement timeline
        improvement_timeline = analyzer._generate_application_timeline(target_crop)
        
        return {
            'soil_health_status': analysis,
            'fertilizer_recommendations': fertilizer_plan,
            'improvement_timeline': improvement_timeline,
            'priority_actions': analysis['recommendations'][:3],  # Top 3 priorities
            'monitoring_plan': {
                'test_frequency': 'Every 6 months during improvement period',
                'key_indicators': ['pH', 'organic_carbon', 'major_nutrients'],
                'success_metrics': 'Target overall score > 7.5'
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating soil improvement recommendations: {str(e)}")
        raise

import uuid
