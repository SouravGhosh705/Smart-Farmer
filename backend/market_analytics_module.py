#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Price Tracking and Analytics Module for Smart Farmer Application
Real-time mandi prices, trend analysis, and market intelligence
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)

@dataclass
class MarketAlert:
    """Data class for market alerts"""
    alert_id: str
    crop_name: str
    alert_type: str
    current_price: float
    previous_price: float
    change_percentage: float
    message: str
    location: str
    timestamp: datetime
    recommendations: List[str]

class MarketPriceTracker:
    """Advanced market price tracking and analytics system"""
    
    def __init__(self):
        self.mandi_database = self._load_mandi_database()
        self.crop_market_info = self._load_crop_market_info()
        self.price_history = self._load_price_history()
        self.alert_thresholds = self._load_alert_thresholds()
        os.makedirs('static/market_data', exist_ok=True)
    
    def _load_mandi_database(self) -> Dict:
        """Load mandi (market) database for price tracking"""
        return {
            'uttar_pradesh': {
                'lucknow': {
                    'mandi_code': 'UP_LKO_001',
                    'major_crops': ['wheat', 'rice', 'gram', 'mustard'],
                    'market_days': ['monday', 'wednesday', 'friday'],
                    'contact': '+91-522-2345678'
                },
                'kanpur': {
                    'mandi_code': 'UP_KNP_002',
                    'major_crops': ['wheat', 'gram', 'potato', 'sugarcane'],
                    'market_days': ['tuesday', 'thursday', 'saturday'],
                    'contact': '+91-512-2345678'
                }
            },
            'punjab': {
                'amritsar': {
                    'mandi_code': 'PB_ASR_001',
                    'major_crops': ['wheat', 'rice', 'cotton', 'maize'],
                    'market_days': ['monday', 'wednesday', 'friday'],
                    'contact': '+91-183-2345678'
                },
                'ludhiana': {
                    'mandi_code': 'PB_LDH_002',
                    'major_crops': ['wheat', 'rice', 'cotton'],
                    'market_days': ['tuesday', 'thursday', 'saturday'],
                    'contact': '+91-161-2345678'
                }
            },
            'maharashtra': {
                'mumbai': {
                    'mandi_code': 'MH_MUM_001',
                    'major_crops': ['cotton', 'sugarcane', 'soybean', 'gram'],
                    'market_days': ['daily'],
                    'contact': '+91-22-2345678'
                },
                'pune': {
                    'mandi_code': 'MH_PUN_002',
                    'major_crops': ['cotton', 'sugarcane', 'wheat'],
                    'market_days': ['daily'],
                    'contact': '+91-20-2345678'
                }
            }
        }
    
    def _load_crop_market_info(self) -> Dict:
        """Load comprehensive crop market information"""
        return {
            'wheat': {
                'msp_2024': 2275,  # Minimum Support Price
                'average_market_price': 2400,
                'price_volatility': 'low',
                'seasonal_pattern': {
                    'harvest_season': {'months': [4, 5], 'price_trend': 'declining'},
                    'lean_season': {'months': [8, 9, 10], 'price_trend': 'rising'},
                    'sowing_season': {'months': [11, 12], 'price_trend': 'stable'}
                },
                'major_consuming_states': ['Uttar Pradesh', 'Punjab', 'Haryana', 'Rajasthan'],
                'export_potential': 'high',
                'storage_life': '12-24 months',
                'quality_parameters': ['protein_content', 'gluten_strength', 'test_weight']
            },
            'rice': {
                'msp_2024': 2320,
                'average_market_price': 2500,
                'price_volatility': 'medium',
                'seasonal_pattern': {
                    'harvest_season': {'months': [10, 11], 'price_trend': 'declining'},
                    'lean_season': {'months': [6, 7, 8], 'price_trend': 'rising'},
                    'sowing_season': {'months': [6, 7], 'price_trend': 'stable'}
                },
                'major_consuming_states': ['West Bengal', 'Uttar Pradesh', 'Bihar', 'Odisha'],
                'export_potential': 'very_high',
                'storage_life': '12-18 months',
                'quality_parameters': ['grain_length', 'moisture_content', 'broken_percentage']
            },
            'cotton': {
                'msp_2024': 6620,
                'average_market_price': 7200,
                'price_volatility': 'high',
                'seasonal_pattern': {
                    'harvest_season': {'months': [10, 11, 12], 'price_trend': 'declining'},
                    'lean_season': {'months': [6, 7, 8], 'price_trend': 'rising'},
                    'sowing_season': {'months': [5, 6], 'price_trend': 'stable'}
                },
                'major_consuming_states': ['Gujarat', 'Maharashtra', 'Telangana', 'Karnataka'],
                'export_potential': 'high',
                'storage_life': '6-12 months',
                'quality_parameters': ['staple_length', 'micronaire', 'trash_content']
            },
            'sugarcane': {
                'msp_2024': 340,  # Per quintal
                'average_market_price': 380,
                'price_volatility': 'low',
                'seasonal_pattern': {
                    'harvest_season': {'months': [1, 2, 3, 4], 'price_trend': 'stable'},
                    'lean_season': {'months': [8, 9], 'price_trend': 'stable'},
                    'growing_season': {'months': [5, 6, 7], 'price_trend': 'stable'}
                },
                'major_consuming_states': ['Uttar Pradesh', 'Maharashtra', 'Karnataka'],
                'export_potential': 'low',
                'storage_life': '1-2 days (fresh)',
                'quality_parameters': ['sucrose_content', 'purity', 'fiber_content']
            },
            'gram': {
                'msp_2024': 5440,
                'average_market_price': 5800,
                'price_volatility': 'medium',
                'seasonal_pattern': {
                    'harvest_season': {'months': [3, 4], 'price_trend': 'declining'},
                    'lean_season': {'months': [8, 9, 10], 'price_trend': 'rising'},
                    'sowing_season': {'months': [11, 12], 'price_trend': 'stable'}
                },
                'major_consuming_states': ['Madhya Pradesh', 'Rajasthan', 'Maharashtra'],
                'export_potential': 'medium',
                'storage_life': '6-12 months',
                'quality_parameters': ['size_grade', 'moisture_content', 'damaged_percentage']
            },
            'soybean': {
                'msp_2024': 4600,
                'average_market_price': 4900,
                'price_volatility': 'high',
                'seasonal_pattern': {
                    'harvest_season': {'months': [10, 11], 'price_trend': 'declining'},
                    'lean_season': {'months': [5, 6, 7], 'price_trend': 'rising'},
                    'sowing_season': {'months': [6, 7], 'price_trend': 'volatile'}
                },
                'major_consuming_states': ['Madhya Pradesh', 'Maharashtra', 'Rajasthan'],
                'export_potential': 'medium',
                'storage_life': '6-8 months',
                'quality_parameters': ['oil_content', 'protein_content', 'damaged_percentage']
            }
        }
    
    def _load_price_history(self) -> Dict:
        """Load historical price data"""
        # In real implementation, this would connect to actual price databases
        # For now, generating simulated historical data based on real patterns
        
        price_history = {}
        current_date = datetime.now()
        
        for crop, info in self.crop_market_info.items():
            base_price = info['average_market_price']
            volatility = {'low': 0.05, 'medium': 0.10, 'high': 0.15}[info['price_volatility']]
            
            # Generate 365 days of price history
            prices = []
            for i in range(365):
                date = current_date - timedelta(days=i)
                
                # Apply seasonal patterns
                month = date.month
                seasonal_factor = self._get_seasonal_price_factor(crop, month)
                
                # Add random daily fluctuation
                daily_factor = 1 + np.random.normal(0, volatility)
                
                # Calculate price
                price = base_price * seasonal_factor * daily_factor
                price = max(price, info['msp_2024'])  # Don't go below MSP
                
                prices.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'price': round(price, 2),
                    'volume': np.random.randint(100, 2000)  # Simulated volume
                })
            
            price_history[crop] = sorted(prices, key=lambda x: x['date'])
        
        return price_history
    
    def _get_seasonal_price_factor(self, crop: str, month: int) -> float:
        """Get seasonal price adjustment factor"""
        if crop not in self.crop_market_info:
            return 1.0
        
        seasonal_info = self.crop_market_info[crop]['seasonal_pattern']
        
        for season, data in seasonal_info.items():
            if month in data['months']:
                if data['price_trend'] == 'rising':
                    return 1.15
                elif data['price_trend'] == 'declining':
                    return 0.90
                else:
                    return 1.0
        
        return 1.0
    
    def _load_alert_thresholds(self) -> Dict:
        """Load market alert thresholds"""
        return {
            'price_change': {
                'significant_increase': 5.0,  # % increase
                'significant_decrease': -5.0,  # % decrease
                'major_increase': 10.0,
                'major_decrease': -10.0
            },
            'volume_change': {
                'high_volume': 150,  # % of average
                'low_volume': 50     # % of average
            }
        }
    
    def get_current_mandi_prices(self, state: str, city: str = None, crops: List[str] = None) -> Dict:
        """Get current mandi prices for location"""
        try:
            # In real implementation, this would call actual mandi APIs
            # For now, generating realistic price data
            
            prices = {}
            target_crops = crops or list(self.crop_market_info.keys())
            
            for crop in target_crops:
                if crop in self.crop_market_info:
                    base_info = self.crop_market_info[crop]
                    
                    # Get current price with daily fluctuation
                    base_price = base_info['average_market_price']
                    volatility = {'low': 0.02, 'medium': 0.05, 'high': 0.08}[base_info['price_volatility']]
                    
                    # Current market price
                    daily_factor = 1 + np.random.normal(0, volatility)
                    current_price = base_price * daily_factor
                    current_price = max(current_price, base_info['msp_2024'])
                    
                    # Get yesterday's price for comparison
                    yesterday_factor = 1 + np.random.normal(0, volatility)
                    yesterday_price = base_price * yesterday_factor
                    
                    # Calculate change
                    price_change = current_price - yesterday_price
                    change_percentage = (price_change / yesterday_price) * 100
                    
                    prices[crop] = {
                        'crop_name': crop.title(),
                        'current_price': round(current_price, 2),
                        'previous_price': round(yesterday_price, 2),
                        'change_amount': round(price_change, 2),
                        'change_percentage': round(change_percentage, 2),
                        'msp': base_info['msp_2024'],
                        'price_trend': 'rising' if change_percentage > 1 else 'falling' if change_percentage < -1 else 'stable',
                        'market_status': self._get_market_status(current_price, base_info['msp_2024'], change_percentage),
                        'volume_traded': np.random.randint(500, 5000),  # Simulated volume
                        'quality_premium': self._get_quality_premium(crop),
                        'last_updated': datetime.now().isoformat()
                    }
            
            # Get mandi information
            mandi_info = self._get_mandi_info(state, city)
            
            return {
                'location': {
                    'state': state.title(),
                    'city': city.title() if city else None
                },
                'mandi_info': mandi_info,
                'prices': prices,
                'market_summary': self._generate_market_summary(prices),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting mandi prices: {str(e)}")
            raise
    
    def _get_market_status(self, current_price: float, msp: float, change_percentage: float) -> str:
        """Determine market status based on price and MSP"""
        price_ratio = current_price / msp
        
        if price_ratio >= 1.2 and change_percentage > 2:
            return 'very_bullish'
        elif price_ratio >= 1.1 and change_percentage > 0:
            return 'bullish'
        elif price_ratio >= 1.05:
            return 'stable_above_msp'
        elif price_ratio >= 1.0:
            return 'at_msp'
        else:
            return 'below_msp'
    
    def _get_quality_premium(self, crop: str) -> Dict:
        """Get quality-based price premiums"""
        quality_premiums = {
            'wheat': {
                'premium_grade': {'premium': 5.0, 'criteria': 'Protein >12%, Test weight >78kg/hl'},
                'standard_grade': {'premium': 0.0, 'criteria': 'Standard specifications'},
                'below_grade': {'premium': -3.0, 'criteria': 'Below standard quality'}
            },
            'rice': {
                'premium_grade': {'premium': 8.0, 'criteria': 'Long grain, <5% broken'},
                'standard_grade': {'premium': 0.0, 'criteria': 'Standard specifications'},
                'below_grade': {'premium': -5.0, 'criteria': 'High broken percentage'}
            },
            'cotton': {
                'premium_grade': {'premium': 10.0, 'criteria': 'Staple length >28mm'},
                'standard_grade': {'premium': 0.0, 'criteria': 'Standard specifications'},
                'below_grade': {'premium': -8.0, 'criteria': 'Short staple, high trash'}
            }
        }
        
        return quality_premiums.get(crop, {
            'standard_grade': {'premium': 0.0, 'criteria': 'Standard market grade'}
        })
    
    def _get_mandi_info(self, state: str, city: str = None) -> Dict:
        """Get mandi information for location"""
        state_lower = state.lower().replace(' ', '_')
        
        if state_lower in self.mandi_database:
            if city:
                city_lower = city.lower()
                for mandi_city, info in self.mandi_database[state_lower].items():
                    if city_lower in mandi_city:
                        return {
                            'mandi_name': f"{city.title()} Mandi",
                            'mandi_code': info['mandi_code'],
                            'major_crops': info['major_crops'],
                            'market_days': info['market_days'],
                            'contact': info['contact'],
                            'status': 'active'
                        }
            
            # Return first mandi in state if city not found
            first_mandi = list(self.mandi_database[state_lower].values())[0]
            return {
                'mandi_name': f"{state.title()} Regional Mandi",
                'mandi_code': first_mandi['mandi_code'],
                'major_crops': first_mandi['major_crops'],
                'market_days': first_mandi['market_days'],
                'contact': first_mandi['contact'],
                'status': 'regional'
            }
        
        # Default mandi info
        return {
            'mandi_name': f"{state.title()} Local Market",
            'mandi_code': f"{state[:2].upper()}_LOCAL_001",
            'major_crops': ['wheat', 'rice', 'gram'],
            'market_days': ['monday', 'wednesday', 'friday'],
            'contact': 'Contact local agriculture department',
            'status': 'estimated'
        }
    
    def _generate_market_summary(self, prices: Dict) -> Dict:
        """Generate market summary from current prices"""
        if not prices:
            return {}
        
        price_changes = [data['change_percentage'] for data in prices.values()]
        avg_change = sum(price_changes) / len(price_changes)
        
        rising_crops = [crop for crop, data in prices.items() if data['change_percentage'] > 1]
        falling_crops = [crop for crop, data in prices.items() if data['change_percentage'] < -1]
        
        market_sentiment = 'bullish' if avg_change > 1 else 'bearish' if avg_change < -1 else 'mixed'
        
        return {
            'overall_trend': market_sentiment,
            'average_change': round(avg_change, 2),
            'rising_crops': rising_crops,
            'falling_crops': falling_crops,
            'stable_crops': [crop for crop in prices.keys() if crop not in rising_crops and crop not in falling_crops],
            'top_gainer': max(prices.items(), key=lambda x: x[1]['change_percentage'])[0] if prices else None,
            'top_loser': min(prices.items(), key=lambda x: x[1]['change_percentage'])[0] if prices else None
        }
    
    def get_price_forecast(self, crop: str, days: int = 30) -> Dict:
        """Generate price forecast for a crop"""
        try:
            if crop not in self.crop_market_info:
                return {'error': f'Price data not available for crop: {crop}'}
            
            crop_info = self.crop_market_info[crop]
            current_price = crop_info['average_market_price']
            volatility = {'low': 0.05, 'medium': 0.10, 'high': 0.15}[crop_info['price_volatility']]
            
            # Generate forecast
            forecast_dates = []
            forecast_prices = []
            forecast_confidence = []
            
            for i in range(days):
                forecast_date = datetime.now() + timedelta(days=i)
                
                # Apply seasonal trend
                seasonal_factor = self._get_seasonal_price_factor(crop, forecast_date.month)
                
                # Apply random walk with mean reversion
                if i == 0:
                    predicted_price = current_price
                else:
                    # Mean reversion factor
                    mean_reversion = 0.95 + (current_price - forecast_prices[-1]) / current_price * 0.1
                    
                    # Random component
                    random_factor = 1 + np.random.normal(0, volatility / 30)  # Daily volatility
                    
                    predicted_price = forecast_prices[-1] * seasonal_factor * mean_reversion * random_factor
                    predicted_price = max(predicted_price, crop_info['msp_2024'])  # Floor at MSP
                
                # Confidence decreases with time
                confidence = max(0.5, 1.0 - (i / days) * 0.4)
                
                forecast_dates.append(forecast_date.strftime('%Y-%m-%d'))
                forecast_prices.append(round(predicted_price, 2))
                forecast_confidence.append(round(confidence, 2))
            
            # Calculate forecast statistics
            price_range = (min(forecast_prices), max(forecast_prices))
            avg_forecasted_price = sum(forecast_prices) / len(forecast_prices)
            
            # Determine recommendation
            current_vs_forecast = (avg_forecasted_price - current_price) / current_price * 100
            
            if current_vs_forecast > 5:
                recommendation = 'hold_for_better_prices'
                advice = 'Prices expected to rise - consider holding stock if possible'
            elif current_vs_forecast < -5:
                recommendation = 'sell_immediately'
                advice = 'Prices expected to fall - consider selling immediately'
            else:
                recommendation = 'sell_gradually'
                advice = 'Stable prices expected - sell gradually based on need'
            
            return {
                'crop': crop,
                'forecast_period': f"{days} days",
                'current_price': current_price,
                'forecast_data': {
                    'dates': forecast_dates,
                    'prices': forecast_prices,
                    'confidence': forecast_confidence
                },
                'statistics': {
                    'average_forecasted_price': round(avg_forecasted_price, 2),
                    'price_range': price_range,
                    'expected_change': round(current_vs_forecast, 2),
                    'volatility_rating': crop_info['price_volatility']
                },
                'recommendation': {
                    'action': recommendation,
                    'advice': advice,
                    'confidence': round(sum(forecast_confidence) / len(forecast_confidence), 2)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating price forecast: {str(e)}")
            raise
    
    def get_market_alerts(self, crops: List[str], location: str = None) -> List[MarketAlert]:
        """Check for market alerts and price opportunities"""
        try:
            alerts = []
            
            for crop in crops:
                if crop not in self.crop_market_info:
                    continue
                
                # Get recent price data
                crop_history = self.price_history.get(crop, [])
                if len(crop_history) < 2:
                    continue
                
                current_data = crop_history[-1]
                previous_data = crop_history[-2]
                
                current_price = current_data['price']
                previous_price = previous_data['price']
                change_percentage = ((current_price - previous_price) / previous_price) * 100
                
                # Check for significant price movements
                thresholds = self.alert_thresholds['price_change']
                
                if change_percentage >= thresholds['major_increase']:
                    alert = MarketAlert(
                        alert_id=str(uuid.uuid4()),
                        crop_name=crop,
                        alert_type='major_price_increase',
                        current_price=current_price,
                        previous_price=previous_price,
                        change_percentage=change_percentage,
                        message=f"Major price increase: {crop.title()} up {change_percentage:.1f}%",
                        location=location or 'General',
                        timestamp=datetime.now(),
                        recommendations=[
                            "Consider selling if you have stock",
                            "Monitor for continued upward trend",
                            "Check quality requirements for premium pricing"
                        ]
                    )
                    alerts.append(alert)
                
                elif change_percentage <= thresholds['major_decrease']:
                    alert = MarketAlert(
                        alert_id=str(uuid.uuid4()),
                        crop_name=crop,
                        alert_type='major_price_decrease',
                        current_price=current_price,
                        previous_price=previous_price,
                        change_percentage=change_percentage,
                        message=f"Major price drop: {crop.title()} down {abs(change_percentage):.1f}%",
                        location=location or 'General',
                        timestamp=datetime.now(),
                        recommendations=[
                            "Hold stock if possible - prices may recover",
                            "Consider forward selling for future harvest",
                            "Explore alternative markets"
                        ]
                    )
                    alerts.append(alert)
                
                # Check MSP alerts
                msp = self.crop_market_info[crop]['msp_2024']
                if current_price <= msp * 1.02:  # Within 2% of MSP
                    alert = MarketAlert(
                        alert_id=str(uuid.uuid4()),
                        crop_name=crop,
                        alert_type='near_msp',
                        current_price=current_price,
                        previous_price=msp,
                        change_percentage=((current_price - msp) / msp) * 100,
                        message=f"{crop.title()} price near MSP: ₹{current_price} (MSP: ₹{msp})",
                        location=location or 'General',
                        timestamp=datetime.now(),
                        recommendations=[
                            "Consider government procurement if below MSP",
                            "Check local APMC mandis for better rates",
                            "Ensure proper grading for MSP benefits"
                        ]
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating market alerts: {str(e)}")
            return []
    
    def get_best_selling_time(self, crop: str, harvest_date: str = None) -> Dict:
        """Determine best selling time for a crop"""
        try:
            if crop not in self.crop_market_info:
                return {'error': f'Market data not available for crop: {crop}'}
            
            crop_info = self.crop_market_info[crop]
            
            # If harvest date provided, use it; otherwise use typical harvest time
            if harvest_date:
                harvest_dt = datetime.strptime(harvest_date, '%Y-%m-%d')
            else:
                # Use typical harvest months
                seasonal_pattern = crop_info['seasonal_pattern']
                harvest_months = seasonal_pattern.get('harvest_season', {}).get('months', [datetime.now().month])
                harvest_dt = datetime(datetime.now().year, harvest_months[0], 15)
            
            # Analyze price patterns for next 12 months
            analysis_results = []
            
            for i in range(12):
                future_date = harvest_dt + timedelta(days=30*i)
                month = future_date.month
                
                # Get seasonal price factor
                seasonal_factor = self._get_seasonal_price_factor(crop, month)
                estimated_price = crop_info['average_market_price'] * seasonal_factor
                
                # Storage costs (increases over time)
                storage_cost_per_month = 50  # Rs per quintal per month
                storage_cost = storage_cost_per_month * i
                
                # Net price after storage costs
                net_price = estimated_price - storage_cost
                
                analysis_results.append({
                    'month': future_date.strftime('%B %Y'),
                    'month_number': i,
                    'estimated_price': round(estimated_price, 2),
                    'storage_cost': storage_cost,
                    'net_price': round(net_price, 2),
                    'seasonal_factor': seasonal_factor
                })
            
            # Find optimal selling time
            best_month = max(analysis_results, key=lambda x: x['net_price'])
            
            # Storage feasibility
            storage_life = crop_info['storage_life']
            max_storage_months = int(storage_life.split('-')[1].split()[0]) if '-' in storage_life else 6
            
            # Filter by storage feasibility
            feasible_options = [r for r in analysis_results if r['month_number'] <= max_storage_months]
            best_feasible = max(feasible_options, key=lambda x: x['net_price'])
            
            # Generate recommendations
            recommendations = []
            
            if best_feasible['month_number'] == 0:
                recommendations.extend([
                    "Sell immediately after harvest",
                    "Current market conditions are favorable",
                    "Avoid storage costs"
                ])
            elif best_feasible['month_number'] <= 3:
                recommendations.extend([
                    f"Store for {best_feasible['month_number']} months",
                    f"Target selling in {best_feasible['month']}",
                    "Ensure proper storage conditions"
                ])
            else:
                recommendations.extend([
                    "Consider long-term storage",
                    "Monitor market conditions closely",
                    "Ensure adequate storage facilities"
                ])
            
            return {
                'crop': crop,
                'harvest_reference_date': harvest_dt.strftime('%Y-%m-%d'),
                'analysis_period': '12 months',
                'current_price': crop_info['average_market_price'],
                'monthly_analysis': analysis_results,
                'optimal_selling_time': {
                    'month': best_month['month'],
                    'estimated_price': best_month['estimated_price'],
                    'net_price': best_month['net_price'],
                    'storage_duration': f"{best_month['month_number']} months"
                },
                'feasible_best_time': {
                    'month': best_feasible['month'],
                    'estimated_price': best_feasible['estimated_price'],
                    'net_price': best_feasible['net_price'],
                    'storage_duration': f"{best_feasible['month_number']} months"
                },
                'recommendations': recommendations,
                'storage_considerations': {
                    'max_storage_period': storage_life,
                    'storage_cost_per_month': f"₹{storage_cost_per_month} per quintal",
                    'quality_parameters': crop_info['quality_parameters']
                },
                'risk_factors': self._get_storage_risk_factors(crop, best_feasible['month_number']),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error determining best selling time: {str(e)}")
            raise
    
    def _get_storage_risk_factors(self, crop: str, storage_months: int) -> List[str]:
        """Get risk factors for storing crop"""
        risks = []
        
        if storage_months > 6:
            risks.extend([
                "Long-term storage increases pest risk",
                "Quality degradation over time",
                "Higher storage and insurance costs"
            ])
        elif storage_months > 3:
            risks.extend([
                "Monitor for pest infestation",
                "Maintain proper moisture levels",
                "Regular quality checks needed"
            ])
        
        # Crop-specific risks
        crop_risks = {
            'rice': ['Rice weevil infestation', 'Moisture absorption'],
            'wheat': ['Khapra beetle risk', 'Fungal growth in high humidity'],
            'cotton': ['Pink bollworm in storage', 'Fiber quality degradation'],
            'gram': ['Pulse beetle infestation', 'Split grain issues']
        }
        
        if crop in crop_risks:
            risks.extend(crop_risks[crop])
        
        return risks
    
    def get_market_intelligence(self, crop: str, location: str = None) -> Dict:
        """Get comprehensive market intelligence for a crop"""
        try:
            if crop not in self.crop_market_info:
                return {'error': f'Market intelligence not available for crop: {crop}'}
            
            # Get current prices and trends
            current_data = self.get_current_mandi_prices('uttar pradesh', 'lucknow', [crop])
            crop_price_data = current_data['prices'].get(crop, {})
            
            # Get price forecast
            forecast_data = self.get_price_forecast(crop, 90)  # 3-month forecast
            
            # Get best selling time
            selling_time_data = self.get_best_selling_time(crop)
            
            # Market competition analysis
            competition_analysis = self._analyze_market_competition(crop)
            
            # Export opportunities
            export_analysis = self._analyze_export_opportunities(crop)
            
            # Risk assessment
            market_risks = self._assess_market_risks(crop)
            
            intelligence = {
                'crop': crop,
                'location': location,
                'current_market_snapshot': crop_price_data,
                'price_forecast': forecast_data,
                'optimal_selling_strategy': selling_time_data,
                'market_competition': competition_analysis,
                'export_opportunities': export_analysis,
                'risk_assessment': market_risks,
                'action_recommendations': self._generate_market_action_plan(crop, crop_price_data, forecast_data),
                'timestamp': datetime.now().isoformat()
            }
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Error generating market intelligence: {str(e)}")
            raise
    
    def _analyze_market_competition(self, crop: str) -> Dict:
        """Analyze market competition for the crop"""
        crop_info = self.crop_market_info[crop]
        
        # Simulated competition data
        major_states = crop_info['major_consuming_states']
        
        competition = {
            'supply_pressure': 'medium',  # Based on production estimates
            'demand_outlook': 'stable',   # Based on consumption patterns
            'import_competition': 'low',  # For most crops in India
            'substitute_crops': self._get_substitute_crops(crop),
            'market_concentration': {
                'top_producing_states': major_states[:3],
                'market_share_top_3': 65  # Approximate percentage
            },
            'seasonal_competition': self._get_seasonal_competition(crop)
        }
        
        return competition
    
    def _get_substitute_crops(self, crop: str) -> List[str]:
        """Get substitute crops that compete in the market"""
        substitutes = {
            'wheat': ['rice', 'maize', 'barley'],
            'rice': ['wheat', 'maize'],
            'cotton': ['synthetic_fibers', 'jute'],
            'sugarcane': ['sugar_beet', 'imported_sugar'],
            'soybean': ['groundnut', 'sunflower', 'mustard'],
            'gram': ['other_pulses', 'imported_pulses']
        }
        
        return substitutes.get(crop, [])
    
    def _get_seasonal_competition(self, crop: str) -> Dict:
        """Get seasonal competition information"""
        # This would analyze when competing crops enter the market
        return {
            'peak_competition_months': self._get_harvest_months_competitors(crop),
            'low_competition_months': self._get_lean_season_months(crop),
            'competition_intensity': 'medium'  # Simplified
        }
    
    def _get_harvest_months_competitors(self, crop: str) -> List[int]:
        """Get months when competing crops are harvested"""
        crop_harvest_months = {
            'wheat': [4, 5],
            'rice': [10, 11],
            'cotton': [10, 11, 12],
            'gram': [3, 4],
            'soybean': [10, 11]
        }
        
        return crop_harvest_months.get(crop, [])
    
    def _get_lean_season_months(self, crop: str) -> List[int]:
        """Get lean season months with less competition"""
        harvest_months = self._get_harvest_months_competitors(crop)
        lean_months = []
        
        for month in range(1, 13):
            if month not in harvest_months:
                # Check if it's 4-6 months after harvest (typical lean season)
                months_after_harvest = min((month - h) % 12 for h in harvest_months)
                if 4 <= months_after_harvest <= 8:
                    lean_months.append(month)
        
        return lean_months
    
    def _analyze_export_opportunities(self, crop: str) -> Dict:
        """Analyze export opportunities for the crop"""
        crop_info = self.crop_market_info[crop]
        export_potential = crop_info.get('export_potential', 'low')
        
        # Simulated export data
        export_data = {
            'export_potential': export_potential,
            'major_export_destinations': self._get_export_destinations(crop),
            'export_price_premium': self._get_export_price_premium(crop),
            'export_requirements': self._get_export_requirements(crop),
            'export_season': self._get_export_season(crop)
        }
        
        return export_data
    
    def _get_export_destinations(self, crop: str) -> List[str]:
        """Get major export destinations for crop"""
        destinations = {
            'rice': ['Saudi Arabia', 'Iran', 'Iraq', 'UAE', 'Yemen'],
            'wheat': ['Bangladesh', 'Nepal', 'UAE', 'Qatar'],
            'cotton': ['China', 'Bangladesh', 'Vietnam', 'Egypt'],
            'gram': ['Turkey', 'Canada', 'UAE'],
            'soybean': ['Japan', 'Thailand', 'South Korea']
        }
        
        return destinations.get(crop, ['Various Countries'])
    
    def _get_export_price_premium(self, crop: str) -> float:
        """Get price premium for export quality"""
        premiums = {
            'rice': 15.0,  # % premium over domestic price
            'wheat': 10.0,
            'cotton': 20.0,
            'gram': 12.0,
            'soybean': 8.0
        }
        
        return premiums.get(crop, 5.0)
    
    def _get_export_requirements(self, crop: str) -> List[str]:
        """Get export quality requirements"""
        requirements = {
            'rice': ['Moisture <14%', 'Broken <5%', 'Foreign matter <1%'],
            'wheat': ['Protein >12%', 'Test weight >78kg/hl', 'Moisture <14%'],
            'cotton': ['Staple length >28mm', 'Micronaire 3.5-4.9', 'Trash <4%'],
            'gram': ['Uniform size', 'Moisture <12%', 'Damage <2%']
        }
        
        return requirements.get(crop, ['Standard export quality'])
    
    def _get_export_season(self, crop: str) -> str:
        """Get typical export season for crop"""
        seasons = {
            'rice': 'October to March',
            'wheat': 'April to September',
            'cotton': 'November to April',
            'gram': 'April to August'
        }
        
        return seasons.get(crop, 'Year round')
    
    def _assess_market_risks(self, crop: str) -> Dict:
        """Assess various market risks for the crop"""
        risks = {
            'price_volatility': self.crop_market_info[crop]['price_volatility'],
            'seasonal_risk': self._assess_seasonal_risk(crop),
            'quality_risk': self._assess_quality_risk(crop),
            'storage_risk': self._assess_storage_risk(crop),
            'policy_risk': self._assess_policy_risk(crop),
            'weather_risk': self._assess_weather_market_risk(crop),
            'overall_risk_rating': 'medium'  # Calculated based on individual risks
        }
        
        # Calculate overall risk
        risk_scores = {
            'low': 1, 'medium': 2, 'high': 3
        }
        
        individual_risks = [
            risks['seasonal_risk'],
            risks['quality_risk'],
            risks['storage_risk'],
            risks['weather_risk']
        ]
        
        avg_risk_score = sum(risk_scores.get(risk, 2) for risk in individual_risks) / len(individual_risks)
        
        if avg_risk_score <= 1.5:
            risks['overall_risk_rating'] = 'low'
        elif avg_risk_score >= 2.5:
            risks['overall_risk_rating'] = 'high'
        else:
            risks['overall_risk_rating'] = 'medium'
        
        return risks
    
    def _assess_seasonal_risk(self, crop: str) -> str:
        """Assess seasonal market risks"""
        current_month = datetime.now().month
        crop_info = self.crop_market_info[crop]
        
        # Check if we're in harvest season (high supply = price pressure)
        harvest_months = crop_info['seasonal_pattern'].get('harvest_season', {}).get('months', [])
        
        if current_month in harvest_months:
            return 'high'  # High supply, price pressure
        elif current_month in [(m + 3) % 12 for m in harvest_months]:
            return 'medium'  # Post-harvest adjustment period
        else:
            return 'low'  # Lean season, better prices
    
    def _assess_quality_risk(self, crop: str) -> str:
        """Assess quality-related market risks"""
        quality_sensitive_crops = ['cotton', 'rice', 'wheat']
        
        if crop in quality_sensitive_crops:
            return 'medium'  # Quality affects price significantly
        else:
            return 'low'   # Less quality sensitivity
    
    def _assess_storage_risk(self, crop: str) -> str:
        """Assess storage-related risks"""
        crop_info = self.crop_market_info[crop]
        storage_life = crop_info['storage_life']
        
        if 'days' in storage_life:
            return 'high'  # Perishable crop
        elif '6' in storage_life or '8' in storage_life:
            return 'medium'  # Medium-term storage
        else:
            return 'low'   # Long storage life
    
    def _assess_policy_risk(self, crop: str) -> str:
        """Assess policy and regulatory risks"""
        # Government intervention crops have different risk profiles
        high_intervention_crops = ['wheat', 'rice', 'cotton']
        
        if crop in high_intervention_crops:
            return 'low'  # MSP support reduces risk
        else:
            return 'medium'  # Market-driven pricing
    
    def _assess_weather_market_risk(self, crop: str) -> str:
        """Assess weather-related market risks"""
        weather_sensitive_crops = ['cotton', 'sugarcane', 'rice']
        
        if crop in weather_sensitive_crops:
            return 'high'  # Weather significantly affects production and prices
        else:
            return 'medium'  # Moderate weather sensitivity
    
    def _generate_market_action_plan(self, crop: str, current_data: Dict, forecast_data: Dict) -> List[Dict]:
        """Generate actionable market recommendations"""
        actions = []
        
        if not current_data or not forecast_data:
            return actions
        
        current_price = current_data.get('current_price', 0)
        change_percentage = current_data.get('change_percentage', 0)
        
        forecast_trend = forecast_data.get('recommendation', {}).get('action', 'sell_gradually')
        
        # Immediate actions
        if change_percentage > 5:
            actions.append({
                'priority': 'immediate',
                'action': 'Consider Selling',
                'description': f"Price has increased {change_percentage:.1f}% - good selling opportunity",
                'timeline': 'Within 2-3 days',
                'expected_benefit': 'Capture current high prices'
            })
        
        elif change_percentage < -5:
            actions.append({
                'priority': 'immediate',
                'action': 'Hold Stock',
                'description': f"Price has dropped {abs(change_percentage):.1f}% - wait for recovery",
                'timeline': 'Monitor for 1 week',
                'expected_benefit': 'Avoid selling at low prices'
            })
        
        # Strategic actions based on forecast
        if forecast_trend == 'hold_for_better_prices':
            actions.append({
                'priority': 'strategic',
                'action': 'Plan Storage',
                'description': 'Prices expected to rise - prepare for medium-term storage',
                'timeline': '1-3 months',
                'expected_benefit': 'Higher selling prices'
            })
        
        elif forecast_trend == 'sell_immediately':
            actions.append({
                'priority': 'urgent',
                'action': 'Accelerate Sales',
                'description': 'Prices expected to decline - sell current stock quickly',
                'timeline': 'Within 1 week',
                'expected_benefit': 'Avoid future price decline'
            })
        
        # Quality improvement actions
        actions.append({
            'priority': 'ongoing',
            'action': 'Quality Enhancement',
            'description': 'Focus on quality to get premium prices',
            'timeline': 'Continuous',
            'expected_benefit': '5-15% price premium'
        })
        
        # Market diversification
        actions.append({
            'priority': 'strategic',
            'action': 'Market Diversification',
            'description': 'Explore multiple buyers and markets',
            'timeline': 'Ongoing',
            'expected_benefit': 'Better price realization'
        })
        
        return actions[:5]  # Return top 5 action items

class CommodityMarketService:
    """High-level service for commodity market operations"""
    
    def __init__(self):
        self.price_tracker = MarketPriceTracker()
        self.commodity_exchanges = self._load_commodity_exchanges()
    
    def _load_commodity_exchanges(self) -> Dict:
        """Load commodity exchange information"""
        return {
            'ncdex': {
                'name': 'National Commodity & Derivatives Exchange',
                'crops': ['wheat', 'gram', 'turmeric', 'coriander', 'jeera'],
                'trading_hours': '9:00 AM to 5:00 PM',
                'settlement': 'daily'
            },
            'mcx': {
                'name': 'Multi Commodity Exchange',
                'crops': ['cotton', 'cardamom', 'mentha_oil'],
                'trading_hours': '9:00 AM to 11:30 PM',
                'settlement': 'daily'
            },
            'icex': {
                'name': 'Indian Commodity Exchange',
                'crops': ['diamond', 'gold', 'silver'],  # Non-agri focus
                'trading_hours': '9:00 AM to 11:30 PM',
                'settlement': 'daily'
            }
        }
    
    def get_comprehensive_market_report(self, crops: List[str], location: str = None) -> Dict:
        """Generate comprehensive market report"""
        try:
            report = {
                'report_id': str(uuid.uuid4()),
                'generation_date': datetime.now().isoformat(),
                'location': location,
                'crops_analyzed': crops,
                'market_overview': {},
                'individual_crop_analysis': {},
                'cross_crop_insights': {},
                'recommendations': {}
            }
            
            # Get data for each crop
            all_price_data = {}
            all_forecasts = {}
            
            for crop in crops:
                try:
                    # Get current prices
                    current_data = self.price_tracker.get_current_mandi_prices('uttar pradesh', 'lucknow', [crop])
                    price_data = current_data['prices'].get(crop, {})
                    
                    # Get forecast
                    forecast = self.price_tracker.get_price_forecast(crop, 30)
                    
                    # Get market intelligence
                    intelligence = self.price_tracker.get_market_intelligence(crop, location)
                    
                    all_price_data[crop] = price_data
                    all_forecasts[crop] = forecast
                    
                    report['individual_crop_analysis'][crop] = {
                        'current_market_data': price_data,
                        'forecast': forecast,
                        'market_intelligence': intelligence
                    }
                    
                except Exception as e:
                    logger.error(f"Error analyzing {crop}: {str(e)}")
                    continue
            
            # Generate market overview
            report['market_overview'] = self._generate_market_overview(all_price_data)
            
            # Cross-crop insights
            report['cross_crop_insights'] = self._generate_cross_crop_insights(all_price_data, all_forecasts)
            
            # Portfolio recommendations
            report['recommendations'] = self._generate_portfolio_recommendations(crops, all_price_data, all_forecasts)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive market report: {str(e)}")
            raise
    
    def _generate_market_overview(self, all_price_data: Dict) -> Dict:
        """Generate overall market overview"""
        if not all_price_data:
            return {}
        
        # Calculate market statistics
        all_changes = [data.get('change_percentage', 0) for data in all_price_data.values()]
        
        overview = {
            'total_crops_tracked': len(all_price_data),
            'average_price_change': round(sum(all_changes) / len(all_changes), 2),
            'crops_rising': len([c for c in all_changes if c > 1]),
            'crops_falling': len([c for c in all_changes if c < -1]),
            'crops_stable': len([c for c in all_changes if -1 <= c <= 1]),
            'market_sentiment': self._determine_market_sentiment(all_changes),
            'most_volatile_crop': self._get_most_volatile_crop(all_price_data),
            'best_performer': self._get_best_performer(all_price_data),
            'worst_performer': self._get_worst_performer(all_price_data)
        }
        
        return overview
    
    def _determine_market_sentiment(self, price_changes: List[float]) -> str:
        """Determine overall market sentiment"""
        avg_change = sum(price_changes) / len(price_changes)
        positive_crops = len([c for c in price_changes if c > 0])
        total_crops = len(price_changes)
        
        if avg_change > 2 and positive_crops / total_crops > 0.7:
            return 'very_bullish'
        elif avg_change > 0 and positive_crops / total_crops > 0.6:
            return 'bullish'
        elif avg_change < -2 and positive_crops / total_crops < 0.3:
            return 'very_bearish'
        elif avg_change < 0 and positive_crops / total_crops < 0.4:
            return 'bearish'
        else:
            return 'mixed'
    
    def _get_most_volatile_crop(self, price_data: Dict) -> str:
        """Get most volatile crop based on price changes"""
        if not price_data:
            return None
        
        return max(price_data.items(), key=lambda x: abs(x[1].get('change_percentage', 0)))[0]
    
    def _get_best_performer(self, price_data: Dict) -> str:
        """Get best performing crop"""
        if not price_data:
            return None
        
        return max(price_data.items(), key=lambda x: x[1].get('change_percentage', 0))[0]
    
    def _get_worst_performer(self, price_data: Dict) -> str:
        """Get worst performing crop"""
        if not price_data:
            return None
        
        return min(price_data.items(), key=lambda x: x[1].get('change_percentage', 0))[0]
    
    def _generate_cross_crop_insights(self, price_data: Dict, forecasts: Dict) -> Dict:
        """Generate insights across multiple crops"""
        insights = {
            'substitution_opportunities': [],
            'diversification_advice': [],
            'timing_strategies': [],
            'portfolio_optimization': {}
        }
        
        # Substitution opportunities
        for crop1, data1 in price_data.items():
            for crop2, data2 in price_data.items():
                if crop1 != crop2:
                    substitutes1 = self.price_tracker._get_substitute_crops(crop1)
                    if crop2 in substitutes1:
                        price_diff = data2['current_price'] - data1['current_price']
                        if price_diff > data1['current_price'] * 0.1:  # 10% higher
                            insights['substitution_opportunities'].append({
                                'from_crop': crop1,
                                'to_crop': crop2,
                                'price_advantage': round((price_diff / data1['current_price']) * 100, 1),
                                'recommendation': f"Consider switching from {crop1} to {crop2} for better returns"
                            })
        
        # Diversification advice
        volatility_levels = {crop: self.price_tracker.crop_market_info[crop]['price_volatility'] 
                           for crop in price_data.keys() 
                           if crop in self.price_tracker.crop_market_info}
        
        stable_crops = [crop for crop, vol in volatility_levels.items() if vol == 'low']
        volatile_crops = [crop for crop, vol in volatility_levels.items() if vol == 'high']
        
        if len(stable_crops) > 0 and len(volatile_crops) > 0:
            insights['diversification_advice'].append({
                'strategy': 'Risk Balancing',
                'stable_crops': stable_crops,
                'volatile_crops': volatile_crops,
                'recommendation': 'Balance portfolio with both stable and high-return crops'
            })
        
        return insights
    
    def _generate_portfolio_recommendations(self, crops: List[str], price_data: Dict, forecasts: Dict) -> Dict:
        """Generate portfolio-level recommendations"""
        recommendations = {
            'immediate_actions': [],
            'short_term_strategy': [],
            'long_term_planning': [],
            'risk_management': []
        }
        
        # Immediate actions based on current price movements
        for crop, data in price_data.items():
            change_pct = data.get('change_percentage', 0)
            
            if change_pct > 5:
                recommendations['immediate_actions'].append({
                    'crop': crop,
                    'action': 'sell_portion',
                    'description': f"Sell 30-50% of {crop} stock to capture gains",
                    'urgency': 'high'
                })
            elif change_pct < -5:
                recommendations['immediate_actions'].append({
                    'crop': crop,
                    'action': 'hold_and_monitor',
                    'description': f"Hold {crop} stock, monitor for recovery signals",
                    'urgency': 'medium'
                })
        
        # Short-term strategy (1-3 months)
        for crop, forecast in forecasts.items():
            if forecast.get('recommendation', {}).get('action') == 'hold_for_better_prices':
                recommendations['short_term_strategy'].append({
                    'crop': crop,
                    'strategy': 'strategic_holding',
                    'description': f"Store {crop} for 1-3 months for better prices",
                    'expected_benefit': f"₹{forecast['statistics']['expected_change']:.2f} per quintal"
                })
        
        # Long-term planning
        recommendations['long_term_planning'].extend([
            {
                'strategy': 'crop_diversification',
                'description': 'Plan crop mix based on market analysis',
                'implementation': 'Next planting season'
            },
            {
                'strategy': 'market_linkage',
                'description': 'Establish direct buyer relationships',
                'implementation': 'Ongoing'
            }
        ])
        
        # Risk management
        recommendations['risk_management'].extend([
            {
                'strategy': 'price_hedging',
                'description': 'Consider commodity futures for price protection',
                'applicability': 'Large farmers with significant production'
            },
            {
                'strategy': 'quality_focus',
                'description': 'Invest in quality improvement for premium pricing',
                'applicability': 'All farmers'
            }
        ])
        
        return recommendations

# Government API integration (placeholder)
def fetch_agmarknet_prices(state: str, district: str, crop: str) -> Dict:
    """Fetch real-time prices from Government's Agmarknet API"""
    try:
        # This would integrate with actual Agmarknet API
        # API endpoint: https://api.data.gov.in/resource/3daeaeb5-e40c-4afa-aba6-7ee46d37a87e
        
        # For demonstration, returning simulated data structure
        api_data = {
            'state': state,
            'district': district,
            'crop': crop,
            'price': np.random.uniform(2000, 8000),
            'arrival': np.random.randint(100, 1000),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'source': 'Agmarknet',
            'reliability': 'high'
        }
        
        logger.info(f"Fetched Agmarknet data for {crop} in {district}, {state}")
        return api_data
        
    except Exception as e:
        logger.error(f"Error fetching Agmarknet prices: {str(e)}")
        return {}

def fetch_nse_commodity_prices(crop: str) -> Dict:
    """Fetch commodity prices from NSE"""
    try:
        # This would integrate with NSE commodity APIs
        # For demonstration, returning simulated futures data
        
        futures_data = {
            'symbol': f"{crop.upper()}",
            'spot_price': np.random.uniform(2000, 8000),
            'future_price': np.random.uniform(2000, 8000),
            'open_interest': np.random.randint(1000, 10000),
            'volume': np.random.randint(500, 5000),
            'last_updated': datetime.now().isoformat(),
            'source': 'NSE',
            'contract_months': ['JAN', 'FEB', 'MAR', 'APR']
        }
        
        return futures_data
        
    except Exception as e:
        logger.error(f"Error fetching NSE commodity prices: {str(e)}")
        return {}

# Market analytics functions
def get_real_time_market_data(crops: List[str], location: str = None) -> Dict:
    """Get comprehensive real-time market data"""
    try:
        service = CommodityMarketService()
        
        # Get mandi prices
        mandi_data = service.price_tracker.get_current_mandi_prices('uttar pradesh', 'lucknow', crops)
        
        # Get commodity exchange data
        exchange_data = {}
        for crop in crops:
            exchange_data[crop] = fetch_nse_commodity_prices(crop)
        
        # Generate market alerts
        alerts = service.price_tracker.get_market_alerts(crops, location)
        
        return {
            'mandi_prices': mandi_data,
            'commodity_exchange_data': exchange_data,
            'market_alerts': [alert.__dict__ for alert in alerts],
            'market_summary': {
                'total_alerts': len(alerts),
                'critical_alerts': len([a for a in alerts if 'major' in a.alert_type]),
                'opportunities': len([a for a in alerts if 'increase' in a.alert_type])
            },
            'data_sources': ['Local Mandi', 'Agmarknet', 'NSE', 'MCX'],
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting real-time market data: {str(e)}")
        return {'error': str(e)}

def get_market_trends_analysis(crop: str, period_days: int = 90) -> Dict:
    """Analyze market trends for a specific period"""
    try:
        service = CommodityMarketService()
        
        # Get historical data
        price_history = service.price_tracker.price_history.get(crop, [])
        
        if not price_history:
            return {'error': f'No historical data available for {crop}'}
        
        # Filter data for specified period
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_data = [
            entry for entry in price_history 
            if datetime.strptime(entry['date'], '%Y-%m-%d') >= cutoff_date
        ]
        
        if len(recent_data) < 10:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Calculate trend metrics
        prices = [entry['price'] for entry in recent_data]
        dates = [entry['date'] for entry in recent_data]
        volumes = [entry['volume'] for entry in recent_data]
        
        # Basic statistics
        current_price = prices[-1]
        start_price = prices[0]
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        
        # Trend analysis
        overall_change = ((current_price - start_price) / start_price) * 100
        volatility = np.std(prices) / avg_price * 100
        
        # Moving averages
        ma_7 = sum(prices[-7:]) / min(7, len(prices))
        ma_15 = sum(prices[-15:]) / min(15, len(prices))
        ma_30 = sum(prices[-30:]) / min(30, len(prices))
        
        # Trend signals
        trend_signals = []
        if current_price > ma_7 > ma_15 > ma_30:
            trend_signals.append("Strong uptrend - all moving averages aligned")
        elif current_price < ma_7 < ma_15 < ma_30:
            trend_signals.append("Strong downtrend - bearish signal")
        elif current_price > ma_7:
            trend_signals.append("Short-term bullish - price above 7-day average")
        else:
            trend_signals.append("Mixed signals - monitor for clearer direction")
        
        return {
            'crop': crop,
            'analysis_period': f"{period_days} days",
            'price_statistics': {
                'current_price': round(current_price, 2),
                'period_start_price': round(start_price, 2),
                'min_price': round(min_price, 2),
                'max_price': round(max_price, 2),
                'average_price': round(avg_price, 2),
                'overall_change': round(overall_change, 2),
                'volatility_percentage': round(volatility, 2)
            },
            'moving_averages': {
                '7_day': round(ma_7, 2),
                '15_day': round(ma_15, 2),
                '30_day': round(ma_30, 2)
            },
            'trend_analysis': {
                'trend_direction': 'upward' if overall_change > 0 else 'downward',
                'trend_strength': 'strong' if abs(overall_change) > 10 else 'moderate' if abs(overall_change) > 5 else 'weak',
                'volatility_rating': 'high' if volatility > 15 else 'medium' if volatility > 8 else 'low',
                'trend_signals': trend_signals
            },
            'historical_data': {
                'dates': dates[-30:],  # Last 30 days for charting
                'prices': prices[-30:],
                'volumes': volumes[-30:]
            },
            'market_opportunities': self._identify_market_opportunities(current_price, avg_price, overall_change),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {str(e)}")
        return {'error': str(e)}

def _identify_market_opportunities(current_price: float, avg_price: float, overall_change: float) -> List[str]:
    """Identify market opportunities based on price analysis"""
    opportunities = []
    
    price_vs_avg = ((current_price - avg_price) / avg_price) * 100
    
    if price_vs_avg > 10:
        opportunities.append("Price is significantly above average - good selling opportunity")
    elif price_vs_avg < -10:
        opportunities.append("Price is below average - potential buying opportunity for storage")
    
    if overall_change > 15:
        opportunities.append("Strong upward trend - consider partial profit booking")
    elif overall_change < -15:
        opportunities.append("Significant decline - may be near bottom, monitor for reversal")
    
    if -5 <= overall_change <= 5:
        opportunities.append("Price consolidation phase - wait for clear direction")
    
    return opportunities

# Global market service instance
market_service = CommodityMarketService()

def get_market_intelligence_report(crops: List[str], state: str, city: str = None) -> Dict:
    """Get comprehensive market intelligence report"""
    try:
        return market_service.get_comprehensive_market_report(crops, f"{city}, {state}" if city else state)
    except Exception as e:
        logger.error(f"Error generating market intelligence: {str(e)}")
        return {'error': str(e)}

def get_price_alerts_for_crops(crops: List[str], location: str = None) -> Dict:
    """Get price alerts for specified crops"""
    try:
        alerts = market_service.price_tracker.get_market_alerts(crops, location)
        
        return {
            'location': location,
            'crops_monitored': crops,
            'alert_count': len(alerts),
            'alerts': [{
                'alert_id': alert.alert_id,
                'crop_name': alert.crop_name,
                'alert_type': alert.alert_type,
                'current_price': alert.current_price,
                'change_percentage': alert.change_percentage,
                'message': alert.message,
                'recommendations': alert.recommendations,
                'timestamp': alert.timestamp.isoformat()
            } for alert in alerts],
            'summary': {
                'price_increase_alerts': len([a for a in alerts if 'increase' in a.alert_type]),
                'price_decrease_alerts': len([a for a in alerts if 'decrease' in a.alert_type]),
                'msp_alerts': len([a for a in alerts if 'msp' in a.alert_type])
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting price alerts: {str(e)}")
        return {'error': str(e)}
