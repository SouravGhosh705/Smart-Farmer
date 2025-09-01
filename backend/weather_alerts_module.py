#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Weather-based Alerts System for Smart Farmer Application
Provides real-time weather monitoring, alerts, and predictive insights
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WeatherAlert:
    """Data class for weather alerts"""
    alert_id: str
    alert_type: str
    severity: str
    title: str
    message: str
    location: str
    timestamp: datetime
    valid_until: datetime
    recommendations: List[str]
    affected_crops: List[str]

class WeatherAlertsSystem:
    """Advanced weather monitoring and alert system for farmers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.alert_thresholds = self._load_alert_thresholds()
        self.crop_weather_requirements = self._load_crop_weather_requirements()
        self.active_alerts = []
    
    def _load_alert_thresholds(self) -> Dict:
        """Load weather alert thresholds"""
        return {
            'temperature': {
                'heat_wave': {'min': 40, 'severity': 'high'},
                'extreme_heat': {'min': 45, 'severity': 'critical'},
                'cold_wave': {'max': 5, 'severity': 'high'},
                'frost': {'max': 0, 'severity': 'critical'}
            },
            'rainfall': {
                'heavy_rain': {'min': 64.5, 'max': 115.5, 'severity': 'medium'},
                'very_heavy_rain': {'min': 115.5, 'max': 204.4, 'severity': 'high'},
                'extremely_heavy_rain': {'min': 204.4, 'severity': 'critical'},
                'drought_warning': {'max': 2.5, 'days': 15, 'severity': 'medium'}
            },
            'wind': {
                'strong_wind': {'min': 25, 'max': 40, 'severity': 'medium'},
                'very_strong_wind': {'min': 40, 'max': 60, 'severity': 'high'},
                'storm': {'min': 60, 'severity': 'critical'}
            },
            'humidity': {
                'very_low': {'max': 30, 'severity': 'medium'},
                'very_high': {'min': 90, 'severity': 'medium'}
            }
        }
    
    def _load_crop_weather_requirements(self) -> Dict:
        """Load crop-specific weather requirements and vulnerabilities"""
        return {
            'rice': {
                'optimal_temp': (20, 35),
                'optimal_humidity': (70, 90),
                'water_requirement': 'high',
                'vulnerable_to': ['drought', 'extreme_heat', 'strong_wind'],
                'critical_stages': {
                    'transplanting': {'rain_sensitive': True, 'temp_range': (25, 30)},
                    'flowering': {'rain_sensitive': True, 'temp_range': (28, 32)},
                    'grain_filling': {'heat_sensitive': True, 'temp_max': 35}
                }
            },
            'wheat': {
                'optimal_temp': (15, 25),
                'optimal_humidity': (50, 70),
                'water_requirement': 'medium',
                'vulnerable_to': ['heat_wave', 'frost', 'heavy_rain'],
                'critical_stages': {
                    'germination': {'temp_range': (15, 20), 'frost_sensitive': True},
                    'tillering': {'frost_sensitive': True, 'temp_range': (10, 20)},
                    'grain_filling': {'heat_sensitive': True, 'temp_max': 30}
                }
            },
            'cotton': {
                'optimal_temp': (21, 30),
                'optimal_humidity': (50, 80),
                'water_requirement': 'high',
                'vulnerable_to': ['extreme_heat', 'heavy_rain', 'strong_wind'],
                'critical_stages': {
                    'germination': {'temp_range': (25, 30), 'rain_sensitive': False},
                    'flowering': {'temp_range': (25, 32), 'rain_sensitive': True},
                    'boll_development': {'heat_sensitive': True, 'temp_max': 35}
                }
            },
            'maize': {
                'optimal_temp': (18, 27),
                'optimal_humidity': (60, 80),
                'water_requirement': 'medium',
                'vulnerable_to': ['drought', 'extreme_heat', 'strong_wind'],
                'critical_stages': {
                    'germination': {'temp_range': (18, 25), 'moisture_critical': True},
                    'tasseling': {'heat_sensitive': True, 'temp_max': 32},
                    'grain_filling': {'drought_sensitive': True}
                }
            },
            'sugarcane': {
                'optimal_temp': (20, 30),
                'optimal_humidity': (75, 85),
                'water_requirement': 'very_high',
                'vulnerable_to': ['drought', 'frost', 'strong_wind'],
                'critical_stages': {
                    'germination': {'temp_range': (25, 30), 'moisture_critical': True},
                    'grand_growth': {'drought_sensitive': True, 'temp_range': (25, 35)},
                    'maturity': {'frost_sensitive': True}
                }
            }
        }
    
    def get_current_weather(self, city: str, state: str = None) -> Dict:
        """Get current weather data with enhanced details"""
        try:
            location = f"{city},{state},IN" if state else f"{city},IN"
            url = f"{self.base_url}/weather?q={location}&appid={self.api_key}&units=metric"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                weather_data = {
                    'current': {
                        'temperature': float(data['main']['temp']),
                        'feels_like': float(data['main']['feels_like']),
                        'humidity': float(data['main']['humidity']),
                        'pressure': float(data['main']['pressure']),
                        'visibility': float(data.get('visibility', 10000)) / 1000,  # Convert to km
                        'uv_index': self._get_uv_index(data['coord']['lat'], data['coord']['lon']),
                        'weather_main': data['weather'][0]['main'],
                        'weather_description': data['weather'][0]['description'],
                        'wind_speed': float(data.get('wind', {}).get('speed', 0)),
                        'wind_direction': float(data.get('wind', {}).get('deg', 0)),
                        'cloudiness': float(data.get('clouds', {}).get('all', 0)),
                        'rainfall_1h': float(data.get('rain', {}).get('1h', 0)),
                        'rainfall_3h': float(data.get('rain', {}).get('3h', 0))
                    },
                    'location': {
                        'name': data['name'],
                        'country': data['sys']['country'],
                        'latitude': data['coord']['lat'],
                        'longitude': data['coord']['lon'],
                        'timezone': data['timezone']
                    },
                    'sun': {
                        'sunrise': datetime.fromtimestamp(data['sys']['sunrise']),
                        'sunset': datetime.fromtimestamp(data['sys']['sunset'])
                    },
                    'timestamp': datetime.now()
                }
                
                return weather_data
            else:
                raise Exception(f"Weather API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching current weather: {str(e)}")
            raise
    
    def _get_uv_index(self, lat: float, lon: float) -> float:
        """Get UV index for location"""
        try:
            url = f"{self.base_url}/uvi?lat={lat}&lon={lon}&appid={self.api_key}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return float(data.get('value', 5.0))
            else:
                return 5.0  # Default moderate UV
        except Exception:
            return 5.0  # Default fallback
    
    def get_weather_forecast(self, city: str, state: str = None, days: int = 5) -> Dict:
        """Get detailed weather forecast"""
        try:
            location = f"{city},{state},IN" if state else f"{city},IN"
            url = f"{self.base_url}/forecast?q={location}&appid={self.api_key}&units=metric"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                forecast_data = {
                    'location': {
                        'name': data['city']['name'],
                        'country': data['city']['country'],
                        'coordinates': {
                            'lat': data['city']['coord']['lat'],
                            'lon': data['city']['coord']['lon']
                        }
                    },
                    'forecast': []
                }
                
                # Process forecast data (3-hour intervals for 5 days)
                for item in data['list'][:days*8]:  # 8 intervals per day
                    forecast_item = {
                        'datetime': datetime.fromtimestamp(item['dt']),
                        'temperature': {
                            'temp': float(item['main']['temp']),
                            'feels_like': float(item['main']['feels_like']),
                            'temp_min': float(item['main']['temp_min']),
                            'temp_max': float(item['main']['temp_max'])
                        },
                        'humidity': float(item['main']['humidity']),
                        'pressure': float(item['main']['pressure']),
                        'weather': {
                            'main': item['weather'][0]['main'],
                            'description': item['weather'][0]['description'],
                            'icon': item['weather'][0]['icon']
                        },
                        'wind': {
                            'speed': float(item.get('wind', {}).get('speed', 0)),
                            'direction': float(item.get('wind', {}).get('deg', 0))
                        },
                        'cloudiness': float(item.get('clouds', {}).get('all', 0)),
                        'precipitation': {
                            'rain_3h': float(item.get('rain', {}).get('3h', 0)),
                            'snow_3h': float(item.get('snow', {}).get('3h', 0))
                        },
                        'probability_of_precipitation': float(item.get('pop', 0)) * 100
                    }
                    forecast_data['forecast'].append(forecast_item)
                
                return forecast_data
            else:
                raise Exception(f"Weather forecast API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {str(e)}")
            raise
    
    def check_weather_alerts(self, city: str, state: str = None, crops: List[str] = None) -> List[WeatherAlert]:
        """Check for weather alerts and generate farming advisories"""
        try:
            alerts = []
            
            # Get current weather and forecast
            current_weather = self.get_current_weather(city, state)
            forecast = self.get_weather_forecast(city, state)
            
            # Check current conditions
            current_alerts = self._check_current_conditions(current_weather, crops)
            alerts.extend(current_alerts)
            
            # Check forecast conditions
            forecast_alerts = self._check_forecast_conditions(forecast, crops)
            alerts.extend(forecast_alerts)
            
            # Update active alerts
            self.active_alerts = alerts
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking weather alerts: {str(e)}")
            return []
    
    def _check_current_conditions(self, weather_data: Dict, crops: List[str] = None) -> List[WeatherAlert]:
        """Check current weather conditions for alerts"""
        alerts = []
        current = weather_data['current']
        location = weather_data['location']['name']
        
        # Temperature alerts
        temp = current['temperature']
        if temp >= self.alert_thresholds['temperature']['extreme_heat']['min']:
            alert = WeatherAlert(
                alert_id=f"temp_extreme_{datetime.now().strftime('%Y%m%d_%H%M')}",
                alert_type="extreme_heat",
                severity="critical",
                title="🔥 Extreme Heat Alert",
                message=f"Temperature is {temp}°C - extremely high for crop safety",
                location=location,
                timestamp=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=6),
                recommendations=[
                    "Increase irrigation frequency immediately",
                    "Provide shade nets for vulnerable crops",
                    "Harvest mature crops early if possible",
                    "Apply mulching to conserve soil moisture"
                ],
                affected_crops=self._get_heat_sensitive_crops(crops)
            )
            alerts.append(alert)
        
        elif temp >= self.alert_thresholds['temperature']['heat_wave']['min']:
            alert = WeatherAlert(
                alert_id=f"temp_heat_{datetime.now().strftime('%Y%m%d_%H%M')}",
                alert_type="heat_wave",
                severity="high",
                title="🌡️ Heat Wave Warning",
                message=f"High temperature detected: {temp}°C",
                location=location,
                timestamp=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=12),
                recommendations=[
                    "Increase watering frequency",
                    "Avoid field operations during peak hours (11 AM - 4 PM)",
                    "Monitor crops for heat stress symptoms"
                ],
                affected_crops=self._get_heat_sensitive_crops(crops)
            )
            alerts.append(alert)
        
        elif temp <= self.alert_thresholds['temperature']['frost']['max']:
            alert = WeatherAlert(
                alert_id=f"temp_frost_{datetime.now().strftime('%Y%m%d_%H%M')}",
                alert_type="frost",
                severity="critical",
                title="❄️ Frost Alert",
                message=f"Freezing temperature: {temp}°C - immediate crop protection needed",
                location=location,
                timestamp=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=8),
                recommendations=[
                    "Cover sensitive crops immediately",
                    "Light smudge fires if safe to do so",
                    "Harvest mature crops before damage",
                    "Check irrigation systems for freezing"
                ],
                affected_crops=self._get_frost_sensitive_crops(crops)
            )
            alerts.append(alert)
        
        # Wind alerts
        wind_speed = current['wind_speed'] * 3.6  # Convert m/s to km/h
        if wind_speed >= self.alert_thresholds['wind']['storm']['min']:
            alert = WeatherAlert(
                alert_id=f"wind_storm_{datetime.now().strftime('%Y%m%d_%H%M')}",
                alert_type="storm",
                severity="critical",
                title="💨 Storm Warning",
                message=f"Very strong winds: {wind_speed:.1f} km/h - secure crops and equipment",
                location=location,
                timestamp=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=6),
                recommendations=[
                    "Secure or harvest tall crops immediately",
                    "Remove or secure farm equipment",
                    "Check structural integrity of greenhouses",
                    "Avoid field operations until winds subside"
                ],
                affected_crops=self._get_wind_sensitive_crops(crops)
            )
            alerts.append(alert)
        
        # Humidity alerts
        humidity = current['humidity']
        if humidity <= self.alert_thresholds['humidity']['very_low']['max']:
            alert = WeatherAlert(
                alert_id=f"humidity_low_{datetime.now().strftime('%Y%m%d_%H%M')}",
                alert_type="low_humidity",
                severity="medium",
                title="🏜️ Low Humidity Alert",
                message=f"Very low humidity: {humidity}% - increased water stress risk",
                location=location,
                timestamp=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=24),
                recommendations=[
                    "Increase irrigation frequency",
                    "Apply mulching to retain soil moisture",
                    "Monitor plants for wilting",
                    "Consider misting for sensitive crops"
                ],
                affected_crops=crops or ['all_crops']
            )
            alerts.append(alert)
        
        elif humidity >= self.alert_thresholds['humidity']['very_high']['min']:
            alert = WeatherAlert(
                alert_id=f"humidity_high_{datetime.now().strftime('%Y%m%d_%H%M')}",
                alert_type="high_humidity",
                severity="medium",
                title="💧 High Humidity Alert",
                message=f"Very high humidity: {humidity}% - fungal disease risk",
                location=location,
                timestamp=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=24),
                recommendations=[
                    "Monitor crops for fungal diseases",
                    "Ensure good air circulation",
                    "Reduce irrigation if soil is wet",
                    "Apply preventive fungicides if necessary"
                ],
                affected_crops=self._get_humidity_sensitive_crops(crops)
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_forecast_conditions(self, forecast_data: Dict, crops: List[str] = None) -> List[WeatherAlert]:
        """Check forecast conditions for upcoming alerts"""
        alerts = []
        location = forecast_data['location']['name']
        
        # Analyze next 48 hours
        forecast_48h = forecast_data['forecast'][:16]  # Next 48 hours (3-hour intervals)
        
        # Check for heavy rainfall forecast
        total_rainfall = sum(item['precipitation']['rain_3h'] for item in forecast_48h)
        max_rainfall_3h = max(item['precipitation']['rain_3h'] for item in forecast_48h)
        
        if max_rainfall_3h >= self.alert_thresholds['rainfall']['extremely_heavy_rain']['min']:
            alert = WeatherAlert(
                alert_id=f"rain_extreme_{datetime.now().strftime('%Y%m%d_%H%M')}",
                alert_type="extreme_rainfall",
                severity="critical",
                title="🌊 Extreme Rainfall Alert",
                message=f"Extremely heavy rainfall expected: {max_rainfall_3h:.1f}mm in 3 hours",
                location=location,
                timestamp=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=48),
                recommendations=[
                    "Ensure proper field drainage immediately",
                    "Harvest mature crops if possible",
                    "Protect stored grain from moisture",
                    "Prepare for potential flooding"
                ],
                affected_crops=crops or ['all_crops']
            )
            alerts.append(alert)
        
        elif total_rainfall >= 100:  # Heavy rainfall over 48 hours
            alert = WeatherAlert(
                alert_id=f"rain_heavy_{datetime.now().strftime('%Y%m%d_%H%M')}",
                alert_type="heavy_rainfall",
                severity="high",
                title="🌧️ Heavy Rainfall Warning",
                message=f"Heavy rainfall expected: {total_rainfall:.1f}mm over next 48 hours",
                location=location,
                timestamp=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=48),
                recommendations=[
                    "Check and clear drainage channels",
                    "Avoid field operations during rain",
                    "Monitor low-lying fields for waterlogging",
                    "Protect fertilizers and pesticides from rain"
                ],
                affected_crops=self._get_flood_sensitive_crops(crops)
            )
            alerts.append(alert)
        
        # Check for temperature extremes in forecast
        max_temp_forecast = max(item['temperature']['temp_max'] for item in forecast_48h)
        min_temp_forecast = min(item['temperature']['temp_min'] for item in forecast_48h)
        
        if max_temp_forecast >= 42:
            alert = WeatherAlert(
                alert_id=f"temp_forecast_high_{datetime.now().strftime('%Y%m%d_%H%M')}",
                alert_type="heat_forecast",
                severity="high",
                title="🔥 Heat Wave Forecast",
                message=f"Extreme heat expected: up to {max_temp_forecast:.1f}°C in next 48 hours",
                location=location,
                timestamp=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=48),
                recommendations=[
                    "Prepare irrigation systems for increased water demand",
                    "Plan field operations for early morning/evening",
                    "Arrange shade protection for sensitive crops",
                    "Monitor livestock for heat stress"
                ],
                affected_crops=self._get_heat_sensitive_crops(crops)
            )
            alerts.append(alert)
        
        if min_temp_forecast <= 2:
            alert = WeatherAlert(
                alert_id=f"temp_forecast_cold_{datetime.now().strftime('%Y%m%d_%H%M')}",
                alert_type="cold_forecast",
                severity="high",
                title="❄️ Cold Wave Forecast",
                message=f"Very cold weather expected: down to {min_temp_forecast:.1f}°C",
                location=location,
                timestamp=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=48),
                recommendations=[
                    "Prepare frost protection measures",
                    "Cover sensitive plants",
                    "Delay planting of warm-season crops",
                    "Check heating systems for livestock"
                ],
                affected_crops=self._get_frost_sensitive_crops(crops)
            )
            alerts.append(alert)
        
        return alerts
    
    def _get_heat_sensitive_crops(self, crops: List[str] = None) -> List[str]:
        """Get crops sensitive to heat stress"""
        heat_sensitive = ['wheat', 'barley', 'gram', 'mustard', 'potato', 'pea']
        if crops:
            return [crop for crop in crops if crop in heat_sensitive]
        return heat_sensitive
    
    def _get_frost_sensitive_crops(self, crops: List[str] = None) -> List[str]:
        """Get crops sensitive to frost"""
        frost_sensitive = ['rice', 'cotton', 'sugarcane', 'banana', 'mango', 'tomato', 'chili']
        if crops:
            return [crop for crop in crops if crop in frost_sensitive]
        return frost_sensitive
    
    def _get_flood_sensitive_crops(self, crops: List[str] = None) -> List[str]:
        """Get crops sensitive to waterlogging"""
        flood_sensitive = ['wheat', 'gram', 'mustard', 'cotton', 'groundnut']
        if crops:
            return [crop for crop in crops if crop in flood_sensitive]
        return flood_sensitive
    
    def _get_wind_sensitive_crops(self, crops: List[str] = None) -> List[str]:
        """Get crops sensitive to strong winds"""
        wind_sensitive = ['cotton', 'sugarcane', 'banana', 'papaya', 'maize']
        if crops:
            return [crop for crop in crops if crop in wind_sensitive]
        return wind_sensitive
    
    def _get_humidity_sensitive_crops(self, crops: List[str] = None) -> List[str]:
        """Get crops sensitive to high humidity (disease-prone)"""
        humidity_sensitive = ['wheat', 'gram', 'mustard', 'tomato', 'potato']
        if crops:
            return [crop for crop in crops if crop in humidity_sensitive]
        return humidity_sensitive
    
    def get_farming_advisories(self, city: str, state: str = None, crops: List[str] = None) -> Dict:
        """Get comprehensive farming advisories based on weather"""
        try:
            current_weather = self.get_current_weather(city, state)
            forecast = self.get_weather_forecast(city, state)
            alerts = self.check_weather_alerts(city, state, crops)
            
            # Generate crop-specific advisories
            crop_advisories = {}
            if crops:
                for crop in crops:
                    crop_advisories[crop] = self._generate_crop_advisory(crop, current_weather, forecast)
            
            # Generate general farming activities recommendation
            activities = self._recommend_farming_activities(current_weather, forecast)
            
            # Generate irrigation schedule
            irrigation_schedule = self._generate_irrigation_schedule(current_weather, forecast, crops)
            
            return {
                'location': {
                    'city': city.title(),
                    'state': state.title() if state else None
                },
                'current_weather': current_weather,
                'forecast_summary': self._summarize_forecast(forecast),
                'alerts': [self._alert_to_dict(alert) for alert in alerts],
                'crop_specific_advisories': crop_advisories,
                'recommended_activities': activities,
                'irrigation_schedule': irrigation_schedule,
                'weekly_outlook': self._generate_weekly_outlook(forecast),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating farming advisories: {str(e)}")
            raise
    
    def _generate_crop_advisory(self, crop: str, current_weather: Dict, forecast: Dict) -> Dict:
        """Generate crop-specific weather advisory"""
        advisory = {
            'crop': crop,
            'current_status': 'normal',
            'recommendations': [],
            'risk_factors': [],
            'optimal_activities': []
        }
        
        if crop not in self.crop_weather_requirements:
            return advisory
        
        crop_req = self.crop_weather_requirements[crop]
        current = current_weather['current']
        
        # Check current temperature suitability
        opt_temp_min, opt_temp_max = crop_req['optimal_temp']
        current_temp = current['temperature']
        
        if current_temp < opt_temp_min:
            advisory['current_status'] = 'suboptimal_cold'
            advisory['recommendations'].append(f"Temperature ({current_temp}°C) is below optimal range for {crop}")
            advisory['risk_factors'].append('cold_stress')
        elif current_temp > opt_temp_max:
            advisory['current_status'] = 'suboptimal_hot'
            advisory['recommendations'].append(f"Temperature ({current_temp}°C) is above optimal range for {crop}")
            advisory['risk_factors'].append('heat_stress')
        else:
            advisory['optimal_activities'].append(f"Good temperature conditions for {crop} field operations")
        
        # Check humidity suitability
        opt_humid_min, opt_humid_max = crop_req['optimal_humidity']
        current_humidity = current['humidity']
        
        if current_humidity < opt_humid_min:
            advisory['recommendations'].append(f"Low humidity ({current_humidity}%) - increase irrigation")
            advisory['risk_factors'].append('water_stress')
        elif current_humidity > opt_humid_max:
            advisory['recommendations'].append(f"High humidity ({current_humidity}%) - monitor for diseases")
            advisory['risk_factors'].append('disease_risk')
        
        # Wind conditions
        wind_speed = current['wind_speed'] * 3.6  # Convert to km/h
        if wind_speed > 30:
            advisory['recommendations'].append(f"Strong winds ({wind_speed:.1f} km/h) - avoid spraying operations")
            advisory['risk_factors'].append('wind_damage')
        elif wind_speed < 5:
            advisory['optimal_activities'].append("Calm conditions - good for pesticide/herbicide application")
        
        return advisory
    
    def _recommend_farming_activities(self, current_weather: Dict, forecast: Dict) -> Dict:
        """Recommend farming activities based on weather"""
        current = current_weather['current']
        activities = {
            'recommended_today': [],
            'avoid_today': [],
            'plan_for_tomorrow': [],
            'weekly_planning': []
        }
        
        # Current conditions analysis
        temp = current['temperature']
        humidity = current['humidity']
        wind_speed = current['wind_speed'] * 3.6
        rainfall = current.get('rainfall_1h', 0)
        
        # Today's recommendations
        if rainfall < 0.1 and wind_speed < 15 and 20 <= temp <= 35:
            activities['recommended_today'].extend([
                "🚜 Field cultivation and land preparation",
                "🌱 Sowing/transplanting operations",
                "💊 Pesticide/herbicide application",
                "🧪 Fertilizer application",
                "🌾 Harvesting mature crops"
            ])
        
        if rainfall > 2.0:
            activities['avoid_today'].extend([
                "❌ All field operations due to rain",
                "❌ Pesticide/fertilizer application",
                "❌ Mechanical harvesting"
            ])
            activities['recommended_today'].extend([
                "✅ Indoor activities (planning, maintenance)",
                "✅ Prepare drainage channels",
                "✅ Check stored grain for moisture"
            ])
        
        if wind_speed > 20:
            activities['avoid_today'].extend([
                "❌ Spraying operations due to wind drift",
                "❌ Working with tall crops"
            ])
        
        if temp > 35:
            activities['avoid_today'].extend([
                "❌ Field operations during peak hours (11 AM - 4 PM)"
            ])
            activities['recommended_today'].extend([
                "✅ Early morning operations (5 AM - 9 AM)",
                "✅ Evening operations (5 PM - 7 PM)"
            ])
        
        # Tomorrow's planning based on forecast
        tomorrow_forecast = forecast['forecast'][8:16]  # Next day's forecast
        tomorrow_rain = sum(item['precipitation']['rain_3h'] for item in tomorrow_forecast)
        tomorrow_max_temp = max(item['temperature']['temp_max'] for item in tomorrow_forecast)
        
        if tomorrow_rain > 10:
            activities['plan_for_tomorrow'].append("🌧️ Heavy rain expected - postpone field operations")
        elif tomorrow_rain < 1:
            activities['plan_for_tomorrow'].append("☀️ Dry weather - good for field work and harvesting")
        
        if tomorrow_max_temp > 38:
            activities['plan_for_tomorrow'].append("🔥 Hot weather expected - plan early morning operations")
        
        return activities
    
    def _generate_irrigation_schedule(self, current_weather: Dict, forecast: Dict, crops: List[str] = None) -> Dict:
        """Generate intelligent irrigation schedule"""
        current = current_weather['current']
        
        # Calculate evapotranspiration estimate
        temp = current['temperature']
        humidity = current['humidity']
        wind_speed = current['wind_speed']
        
        # Simplified ET calculation
        et_factor = (temp / 35) * (1 + wind_speed / 10) * (1 - humidity / 100)
        daily_et = 3 + (et_factor * 2)  # mm/day
        
        # Forecast rainfall for next 7 days
        forecast_7days = forecast['forecast'][:56]  # 7 days * 8 intervals
        total_expected_rain = sum(item['precipitation']['rain_3h'] for item in forecast_7days)
        
        schedule = {
            'current_et_rate': round(daily_et, 1),
            'expected_rainfall_7days': round(total_expected_rain, 1),
            'water_deficit': round(max(0, daily_et * 7 - total_expected_rain), 1),
            'irrigation_needed': daily_et * 7 > total_expected_rain,
            'recommendations': []
        }
        
        if schedule['irrigation_needed']:
            deficit = schedule['water_deficit']
            if deficit > 30:
                schedule['recommendations'].extend([
                    f"🚨 High water deficit ({deficit:.1f}mm) - irrigate immediately",
                    "💧 Deep irrigation required (25-30mm)",
                    "⏰ Schedule irrigation for early morning (5-7 AM)"
                ])
            elif deficit > 15:
                schedule['recommendations'].extend([
                    f"⚠️ Moderate water deficit ({deficit:.1f}mm) - plan irrigation",
                    "💧 Light to medium irrigation (15-20mm)",
                    "⏰ Schedule irrigation for evening (6-8 PM)"
                ])
            else:
                schedule['recommendations'].append("💡 Light irrigation may be beneficial")
        else:
            schedule['recommendations'].append("✅ Sufficient rainfall expected - skip irrigation")
        
        # Crop-specific irrigation timing
        if crops:
            crop_schedules = {}
            for crop in crops:
                if crop in self.crop_weather_requirements:
                    water_req = self.crop_weather_requirements[crop]['water_requirement']
                    
                    if water_req == 'very_high':
                        multiplier = 1.5
                    elif water_req == 'high':
                        multiplier = 1.2
                    elif water_req == 'medium':
                        multiplier = 1.0
                    else:
                        multiplier = 0.8
                    
                    crop_schedules[crop] = {
                        'water_requirement': water_req,
                        'irrigation_frequency': self._calculate_irrigation_frequency(daily_et * multiplier, total_expected_rain),
                        'next_irrigation': self._get_next_irrigation_date(daily_et * multiplier, total_expected_rain)
                    }
            
            schedule['crop_specific'] = crop_schedules
        
        return schedule
    
    def _calculate_irrigation_frequency(self, daily_water_need: float, expected_rain: float) -> str:
        """Calculate irrigation frequency based on water needs"""
        net_water_need = max(0, daily_water_need * 7 - expected_rain)
        
        if net_water_need > 40:
            return "Every 2-3 days"
        elif net_water_need > 20:
            return "Every 4-5 days"
        elif net_water_need > 10:
            return "Weekly"
        else:
            return "As needed (monitor soil moisture)"
    
    def _get_next_irrigation_date(self, daily_water_need: float, expected_rain: float) -> str:
        """Get next recommended irrigation date"""
        if expected_rain > daily_water_need * 3:  # 3 days worth of rain
            return (datetime.now() + timedelta(days=4)).strftime("%Y-%m-%d")
        elif expected_rain > daily_water_need:
            return (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        else:
            return datetime.now().strftime("%Y-%m-%d")  # Today
    
    def _summarize_forecast(self, forecast: Dict) -> Dict:
        """Summarize weather forecast for easy consumption"""
        forecast_items = forecast['forecast'][:40]  # Next 5 days
        
        daily_summaries = []
        current_date = None
        daily_data = []
        
        for item in forecast_items:
            item_date = item['datetime'].date()
            
            if current_date != item_date:
                if daily_data:
                    # Process previous day
                    daily_summaries.append(self._process_daily_data(daily_data, current_date))
                
                current_date = item_date
                daily_data = [item]
            else:
                daily_data.append(item)
        
        # Process last day
        if daily_data:
            daily_summaries.append(self._process_daily_data(daily_data, current_date))
        
        return {
            'daily_summaries': daily_summaries,
            'overall_trend': self._analyze_weather_trend(daily_summaries)
        }
    
    def _process_daily_data(self, daily_data: List[Dict], date) -> Dict:
        """Process weather data for a single day"""
        temps = [item['temperature']['temp'] for item in daily_data]
        humidity_vals = [item['humidity'] for item in daily_data]
        rain_vals = [item['precipitation']['rain_3h'] for item in daily_data]
        wind_vals = [item['wind']['speed'] for item in daily_data]
        
        return {
            'date': date.strftime("%Y-%m-%d"),
            'day_name': date.strftime("%A"),
            'temperature': {
                'min': round(min(temps), 1),
                'max': round(max(temps), 1),
                'avg': round(sum(temps) / len(temps), 1)
            },
            'humidity': {
                'min': round(min(humidity_vals), 1),
                'max': round(max(humidity_vals), 1),
                'avg': round(sum(humidity_vals) / len(humidity_vals), 1)
            },
            'total_rainfall': round(sum(rain_vals), 1),
            'max_wind_speed': round(max(wind_vals) * 3.6, 1),  # Convert to km/h
            'weather_summary': self._get_dominant_weather(daily_data),
            'farming_suitability': self._assess_farming_suitability(daily_data)
        }
    
    def _get_dominant_weather(self, daily_data: List[Dict]) -> str:
        """Get dominant weather condition for the day"""
        weather_counts = {}
        for item in daily_data:
            weather = item['weather']['main']
            weather_counts[weather] = weather_counts.get(weather, 0) + 1
        
        return max(weather_counts, key=weather_counts.get)
    
    def _assess_farming_suitability(self, daily_data: List[Dict]) -> str:
        """Assess overall farming suitability for the day"""
        avg_temp = sum(item['temperature']['temp'] for item in daily_data) / len(daily_data)
        total_rain = sum(item['precipitation']['rain_3h'] for item in daily_data)
        max_wind = max(item['wind']['speed'] for item in daily_data) * 3.6
        
        if total_rain > 15:
            return "Poor - Heavy rainfall expected"
        elif max_wind > 25:
            return "Poor - Strong winds expected"
        elif avg_temp > 38:
            return "Limited - Very hot, early morning operations only"
        elif avg_temp < 10:
            return "Limited - Very cold conditions"
        elif total_rain < 2 and 20 <= avg_temp <= 32 and max_wind < 15:
            return "Excellent - Ideal farming conditions"
        else:
            return "Good - Suitable for most operations"
    
    def _analyze_weather_trend(self, daily_summaries: List[Dict]) -> Dict:
        """Analyze overall weather trend"""
        if len(daily_summaries) < 2:
            return {'trend': 'insufficient_data'}
        
        # Temperature trend
        temps = [day['temperature']['avg'] for day in daily_summaries]
        temp_trend = 'stable'
        if temps[-1] > temps[0] + 3:
            temp_trend = 'warming'
        elif temps[-1] < temps[0] - 3:
            temp_trend = 'cooling'
        
        # Rainfall trend
        rainfall_total = sum(day['total_rainfall'] for day in daily_summaries)
        if rainfall_total > 50:
            rainfall_trend = 'wet_period'
        elif rainfall_total < 5:
            rainfall_trend = 'dry_period'
        else:
            rainfall_trend = 'moderate'
        
        return {
            'temperature_trend': temp_trend,
            'rainfall_trend': rainfall_trend,
            'overall_pattern': self._determine_weather_pattern(temp_trend, rainfall_trend),
            'farming_recommendation': self._get_trend_based_recommendation(temp_trend, rainfall_trend)
        }
    
    def _determine_weather_pattern(self, temp_trend: str, rainfall_trend: str) -> str:
        """Determine overall weather pattern"""
        if temp_trend == 'warming' and rainfall_trend == 'dry_period':
            return 'Hot and Dry - Drought risk'
        elif temp_trend == 'cooling' and rainfall_trend == 'wet_period':
            return 'Cool and Wet - Disease risk'
        elif rainfall_trend == 'wet_period':
            return 'Rainy Period - Waterlogging risk'
        elif rainfall_trend == 'dry_period':
            return 'Dry Period - Irrigation needed'
        else:
            return 'Stable Conditions - Normal operations'
    
    def _get_trend_based_recommendation(self, temp_trend: str, rainfall_trend: str) -> str:
        """Get recommendation based on weather trend"""
        if temp_trend == 'warming' and rainfall_trend == 'dry_period':
            return "Prepare for drought conditions - arrange alternative water sources"
        elif temp_trend == 'cooling' and rainfall_trend == 'wet_period':
            return "Monitor crops for fungal diseases - ensure proper drainage"
        elif rainfall_trend == 'wet_period':
            return "Focus on drainage management and disease prevention"
        elif rainfall_trend == 'dry_period':
            return "Optimize irrigation scheduling and water conservation"
        else:
            return "Continue regular farming operations with weather monitoring"
    
    def _generate_weekly_outlook(self, forecast: Dict) -> Dict:
        """Generate weekly farming outlook"""
        forecast_items = forecast['forecast'][:56]  # 7 days
        
        # Aggregate weekly data
        total_rainfall = sum(item['precipitation']['rain_3h'] for item in forecast_items)
        avg_temp = sum(item['temperature']['temp'] for item in forecast_items) / len(forecast_items)
        max_temp = max(item['temperature']['temp_max'] for item in forecast_items)
        min_temp = min(item['temperature']['temp_min'] for item in forecast_items)
        
        # Determine best days for different activities
        best_days = self._find_best_farming_days(forecast_items)
        
        return {
            'week_summary': {
                'avg_temperature': round(avg_temp, 1),
                'max_temperature': round(max_temp, 1),
                'min_temperature': round(min_temp, 1),
                'total_rainfall': round(total_rainfall, 1),
                'rainy_days': len([item for item in forecast_items if item['precipitation']['rain_3h'] > 2.5])
            },
            'best_days_for_activities': best_days,
            'weekly_recommendations': self._get_weekly_recommendations(total_rainfall, avg_temp, max_temp, min_temp)
        }
    
    def _find_best_farming_days(self, forecast_items: List[Dict]) -> Dict:
        """Find best days for different farming activities"""
        daily_scores = {}
        
        for i in range(0, len(forecast_items), 8):  # Group by day
            day_items = forecast_items[i:i+8]
            date = day_items[0]['datetime'].date()
            
            # Calculate suitability scores
            avg_temp = sum(item['temperature']['temp'] for item in day_items) / len(day_items)
            total_rain = sum(item['precipitation']['rain_3h'] for item in day_items)
            max_wind = max(item['wind']['speed'] for item in day_items) * 3.6
            
            # Scoring for different activities
            scores = {
                'field_work': self._score_field_work(avg_temp, total_rain, max_wind),
                'spraying': self._score_spraying(avg_temp, total_rain, max_wind),
                'harvesting': self._score_harvesting(avg_temp, total_rain, max_wind),
                'irrigation': self._score_irrigation(avg_temp, total_rain)
            }
            
            daily_scores[date.strftime("%A, %B %d")] = scores
        
        # Find best days for each activity
        best_days = {}
        for activity in ['field_work', 'spraying', 'harvesting', 'irrigation']:
            best_day = max(daily_scores.items(), key=lambda x: x[1][activity])
            best_days[activity] = {
                'day': best_day[0],
                'score': best_day[1][activity],
                'suitability': self._get_suitability_rating(best_day[1][activity])
            }
        
        return best_days
    
    def _score_field_work(self, temp: float, rain: float, wind: float) -> float:
        """Score day suitability for field work"""
        score = 10
        
        if rain > 5:
            score -= 8
        elif rain > 2:
            score -= 4
        
        if temp > 35:
            score -= 3
        elif temp < 15:
            score -= 2
        
        if wind > 30:
            score -= 3
        elif wind > 20:
            score -= 1
        
        return max(0, score)
    
    def _score_spraying(self, temp: float, rain: float, wind: float) -> float:
        """Score day suitability for spraying operations"""
        score = 10
        
        if rain > 1:
            score -= 10  # No spraying if rain expected
        
        if wind > 15:
            score -= 8  # Wind drift issues
        elif wind > 10:
            score -= 4
        
        if temp > 30:
            score -= 2  # Evaporation issues
        
        return max(0, score)
    
    def _score_harvesting(self, temp: float, rain: float, wind: float) -> float:
        """Score day suitability for harvesting"""
        score = 10
        
        if rain > 2:
            score -= 9
        elif rain > 0.5:
            score -= 5
        
        if wind > 40:
            score -= 6
        elif wind > 25:
            score -= 3
        
        if temp > 38:
            score -= 2
        
        return max(0, score)
    
    def _score_irrigation(self, temp: float, rain: float) -> float:
        """Score irrigation need for the day"""
        score = 5  # Base score
        
        if rain > 10:
            score = 0  # No irrigation needed
        elif rain > 5:
            score = 2
        else:
            score += (temp - 25) * 0.2  # Higher temp = more irrigation need
        
        return max(0, min(10, score))
    
    def _get_suitability_rating(self, score: float) -> str:
        """Convert numeric score to rating"""
        if score >= 8:
            return "Excellent"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Fair"
        elif score >= 2:
            return "Poor"
        else:
            return "Not Suitable"
    
    def _get_weekly_recommendations(self, total_rain: float, avg_temp: float, max_temp: float, min_temp: float) -> List[str]:
        """Get weekly farming recommendations"""
        recommendations = []
        
        if total_rain > 75:
            recommendations.extend([
                "🌊 Heavy rainfall week - focus on drainage management",
                "🦠 Monitor crops for fungal and bacterial diseases",
                "📦 Protect stored grain from moisture"
            ])
        elif total_rain < 10:
            recommendations.extend([
                "🏜️ Dry week ahead - prepare irrigation systems",
                "💧 Consider water conservation techniques",
                "🌱 Monitor crops for water stress symptoms"
            ])
        
        if max_temp > 40:
            recommendations.extend([
                "🔥 Extreme heat expected - plan protection measures",
                "⏰ Schedule field work for early hours only"
            ])
        
        if min_temp < 10:
            recommendations.extend([
                "❄️ Cold conditions - protect sensitive crops",
                "🔥 Consider frost protection methods"
            ])
        
        if 20 <= avg_temp <= 30 and 10 <= total_rain <= 40:
            recommendations.append("✅ Generally favorable conditions for most farming activities")
        
        return recommendations
    
    def _alert_to_dict(self, alert: WeatherAlert) -> Dict:
        """Convert WeatherAlert object to dictionary"""
        return {
            'alert_id': alert.alert_id,
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'title': alert.title,
            'message': alert.message,
            'location': alert.location,
            'timestamp': alert.timestamp.isoformat(),
            'valid_until': alert.valid_until.isoformat(),
            'recommendations': alert.recommendations,
            'affected_crops': alert.affected_crops
        }

class AgricultureWeatherService:
    """High-level service for agriculture weather monitoring"""
    
    def __init__(self, api_key: str):
        self.weather_system = WeatherAlertsSystem(api_key)
        self.seasonal_calendar = self._load_seasonal_calendar()
    
    def _load_seasonal_calendar(self) -> Dict:
        """Load seasonal farming calendar"""
        return {
            'kharif': {
                'months': [6, 7, 8, 9, 10],
                'crops': ['rice', 'cotton', 'sugarcane', 'maize', 'soybean'],
                'activities': {
                    'june': ['land_preparation', 'sowing'],
                    'july': ['transplanting', 'weeding'],
                    'august': ['fertilizer_application', 'pest_monitoring'],
                    'september': ['disease_management', 'irrigation'],
                    'october': ['harvesting_preparation', 'early_harvest']
                }
            },
            'rabi': {
                'months': [11, 12, 1, 2, 3, 4],
                'crops': ['wheat', 'barley', 'gram', 'mustard', 'pea'],
                'activities': {
                    'november': ['land_preparation', 'sowing'],
                    'december': ['irrigation', 'fertilizer_application'],
                    'january': ['weeding', 'pest_monitoring'],
                    'february': ['disease_management', 'irrigation'],
                    'march': ['grain_filling_monitoring', 'harvest_preparation'],
                    'april': ['harvesting', 'post_harvest_processing']
                }
            },
            'summer': {
                'months': [3, 4, 5, 6],
                'crops': ['fodder_crops', 'vegetables', 'watermelon'],
                'activities': {
                    'march': ['irrigation_system_preparation'],
                    'april': ['heat_tolerant_crop_sowing'],
                    'may': ['intensive_irrigation', 'heat_protection'],
                    'june': ['monsoon_preparation']
                }
            }
        }
    
    def get_comprehensive_weather_advisory(self, city: str, state: str = None, crops: List[str] = None) -> Dict:
        """Get comprehensive weather advisory for farmers"""
        try:
            # Get all weather data
            advisories = self.weather_system.get_farming_advisories(city, state, crops)
            
            # Add seasonal context
            current_month = datetime.now().month
            current_season = self._determine_current_season(current_month)
            seasonal_advice = self._get_seasonal_advice(current_season, current_month)
            
            # Add long-term climate insights
            climate_insights = self._generate_climate_insights(advisories['forecast_summary'])
            
            # Combine all information
            comprehensive_advisory = {
                **advisories,
                'seasonal_context': {
                    'current_season': current_season,
                    'seasonal_advice': seasonal_advice,
                    'recommended_crops': self.seasonal_calendar[current_season]['crops']
                },
                'climate_insights': climate_insights,
                'action_priorities': self._prioritize_actions(advisories['alerts'], seasonal_advice)
            }
            
            return comprehensive_advisory
            
        except Exception as e:
            logger.error(f"Error generating comprehensive weather advisory: {str(e)}")
            raise
    
    def _determine_current_season(self, month: int) -> str:
        """Determine current agricultural season"""
        if month in [6, 7, 8, 9, 10]:
            return 'kharif'
        elif month in [11, 12, 1, 2, 3, 4]:
            return 'rabi'
        else:
            return 'summer'
    
    def _get_seasonal_advice(self, season: str, month: int) -> Dict:
        """Get season-specific farming advice"""
        season_data = self.seasonal_calendar.get(season, {})
        month_name = datetime(2023, month, 1).strftime('%B').lower()
        
        activities = season_data.get('activities', {}).get(month_name, [])
        
        return {
            'season': season,
            'month_activities': activities,
            'recommended_crops': season_data.get('crops', []),
            'general_advice': self._get_seasonal_general_advice(season)
        }
    
    def _get_seasonal_general_advice(self, season: str) -> List[str]:
        """Get general advice for the season"""
        advice = {
            'kharif': [
                "Monitor monsoon patterns closely",
                "Ensure proper field drainage",
                "Watch for pest outbreaks due to humidity",
                "Store fertilizers in dry places"
            ],
            'rabi': [
                "Plan irrigation as no monsoon support",
                "Protect crops from cold waves",
                "Monitor for late blight diseases",
                "Prepare for harvest season"
            ],
            'summer': [
                "Focus on water conservation",
                "Use heat-tolerant crop varieties",
                "Plan irrigation systems maintenance",
                "Prepare for upcoming monsoon"
            ]
        }
        
        return advice.get(season, [])
    
    def _generate_climate_insights(self, forecast_summary: Dict) -> Dict:
        """Generate climate insights from forecast data"""
        daily_summaries = forecast_summary['daily_summaries']
        
        insights = {
            'temperature_analysis': self._analyze_temperature_pattern(daily_summaries),
            'rainfall_analysis': self._analyze_rainfall_pattern(daily_summaries),
            'farming_windows': self._identify_farming_windows(daily_summaries),
            'risk_assessment': self._assess_weather_risks(daily_summaries)
        }
        
        return insights
    
    def _analyze_temperature_pattern(self, daily_summaries: List[Dict]) -> Dict:
        """Analyze temperature patterns"""
        temps = [day['temperature']['avg'] for day in daily_summaries]
        max_temps = [day['temperature']['max'] for day in daily_summaries]
        min_temps = [day['temperature']['min'] for day in daily_summaries]
        
        return {
            'average_temperature': round(sum(temps) / len(temps), 1),
            'temperature_range': round(max(max_temps) - min(min_temps), 1),
            'extreme_heat_days': len([t for t in max_temps if t > 38]),
            'cold_days': len([t for t in min_temps if t < 15]),
            'optimal_days': len([t for t in temps if 20 <= t <= 30]),
            'pattern': 'stable' if max(temps) - min(temps) < 5 else 'variable'
        }
    
    def _analyze_rainfall_pattern(self, daily_summaries: List[Dict]) -> Dict:
        """Analyze rainfall patterns"""
        rainfall_data = [day['total_rainfall'] for day in daily_summaries]
        total_rain = sum(rainfall_data)
        rainy_days = len([r for r in rainfall_data if r > 2.5])
        
        return {
            'total_expected_rainfall': round(total_rain, 1),
            'rainy_days': rainy_days,
            'dry_days': len(daily_summaries) - rainy_days,
            'average_daily_rainfall': round(total_rain / len(daily_summaries), 1),
            'max_daily_rainfall': round(max(rainfall_data), 1),
            'distribution': 'even' if rainy_days > len(daily_summaries) * 0.4 else 'scattered',
            'irrigation_need': 'low' if total_rain > 40 else 'high' if total_rain < 15 else 'medium'
        }
    
    def _identify_farming_windows(self, daily_summaries: List[Dict]) -> Dict:
        """Identify optimal farming windows"""
        windows = {
            'excellent_days': [],
            'good_days': [],
            'poor_days': []
        }
        
        for day in daily_summaries:
            suitability = day['farming_suitability']
            day_name = day['day_name']
            date = day['date']
            
            if suitability in ['Excellent']:
                windows['excellent_days'].append(f"{day_name} ({date})")
            elif suitability in ['Good', 'Limited - Very hot, early morning operations only']:
                windows['good_days'].append(f"{day_name} ({date})")
            else:
                windows['poor_days'].append(f"{day_name} ({date})")
        
        return windows
    
    def _assess_weather_risks(self, daily_summaries: List[Dict]) -> Dict:
        """Assess weather-related risks"""
        risks = {
            'drought_risk': 'low',
            'flood_risk': 'low',
            'heat_stress_risk': 'low',
            'cold_damage_risk': 'low',
            'disease_risk': 'low',
            'overall_risk': 'low'
        }
        
        total_rain = sum(day['total_rainfall'] for day in daily_summaries)
        max_temp = max(day['temperature']['max'] for day in daily_summaries)
        min_temp = min(day['temperature']['min'] for day in daily_summaries)
        avg_humidity = sum(day['humidity']['avg'] for day in daily_summaries) / len(daily_summaries)
        
        # Assess risks
        if total_rain < 10:
            risks['drought_risk'] = 'high'
        elif total_rain > 100:
            risks['flood_risk'] = 'high'
        
        if max_temp > 40:
            risks['heat_stress_risk'] = 'high'
        elif max_temp > 35:
            risks['heat_stress_risk'] = 'medium'
        
        if min_temp < 5:
            risks['cold_damage_risk'] = 'high'
        elif min_temp < 10:
            risks['cold_damage_risk'] = 'medium'
        
        if avg_humidity > 85 and total_rain > 30:
            risks['disease_risk'] = 'high'
        elif avg_humidity > 80:
            risks['disease_risk'] = 'medium'
        
        # Overall risk assessment
        high_risks = [k for k, v in risks.items() if v == 'high' and k != 'overall_risk']
        if len(high_risks) >= 2:
            risks['overall_risk'] = 'high'
        elif len(high_risks) == 1:
            risks['overall_risk'] = 'medium'
        
        return risks
    
    def _prioritize_actions(self, alerts: List[Dict], seasonal_advice: Dict) -> List[Dict]:
        """Prioritize farming actions based on alerts and season"""
        actions = []
        
        # Process alerts by severity
        critical_alerts = [alert for alert in alerts if alert['severity'] == 'critical']
        high_alerts = [alert for alert in alerts if alert['severity'] == 'high']
        
        # Critical actions first
        for alert in critical_alerts:
            actions.append({
                'priority': 'immediate',
                'action': f"Address {alert['alert_type']}",
                'description': alert['message'],
                'recommendations': alert['recommendations'][:2]  # Top 2 recommendations
            })
        
        # High priority actions
        for alert in high_alerts:
            actions.append({
                'priority': 'urgent',
                'action': f"Prepare for {alert['alert_type']}",
                'description': alert['message'],
                'recommendations': alert['recommendations'][:2]
            })
        
        # Seasonal activities
        for activity in seasonal_advice.get('month_activities', []):
            actions.append({
                'priority': 'scheduled',
                'action': activity.replace('_', ' ').title(),
                'description': f"Seasonal activity for {seasonal_advice['season']} season",
                'recommendations': [f"Follow best practices for {activity}"]
            })
        
        return actions[:10]  # Return top 10 prioritized actions

# Global weather service instance
weather_service = None

def get_weather_service(api_key: str = "ff049be539ac8642b805155154206e4c") -> AgricultureWeatherService:
    """Get global weather service instance"""
    global weather_service
    if weather_service is None:
        weather_service = AgricultureWeatherService(api_key)
    return weather_service

def get_weather_alerts_for_location(city: str, state: str = None, crops: List[str] = None) -> Dict:
    """Get weather alerts for a specific location"""
    try:
        service = get_weather_service()
        alerts = service.weather_system.check_weather_alerts(city, state, crops)
        
        return {
            'location': f"{city}, {state}" if state else city,
            'alert_count': len(alerts),
            'alerts': [service.weather_system._alert_to_dict(alert) for alert in alerts],
            'summary': {
                'critical_alerts': len([a for a in alerts if a.severity == 'critical']),
                'high_priority_alerts': len([a for a in alerts if a.severity == 'high']),
                'medium_priority_alerts': len([a for a in alerts if a.severity == 'medium'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting weather alerts: {str(e)}")
        return {
            'error': str(e),
            'location': f"{city}, {state}" if state else city,
            'timestamp': datetime.now().isoformat()
        }

def get_weekly_farming_outlook(city: str, state: str = None, crops: List[str] = None) -> Dict:
    """Get weekly farming outlook with detailed recommendations"""
    try:
        service = get_weather_service()
        return service.get_comprehensive_weather_advisory(city, state, crops)
        
    except Exception as e:
        logger.error(f"Error getting weekly farming outlook: {str(e)}")
        return {
            'error': str(e),
            'location': f"{city}, {state}" if state else city,
            'timestamp': datetime.now().isoformat()
        }
