#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Registry for Lazy Loading and Memory Management
Prevents startup freezing by loading models only when needed
"""

import os
import logging
import threading
from datetime import datetime, timedelta
from joblib import load, dump
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Centralized model management with lazy loading and memory optimization
    """
    
    def __init__(self):
        self._models = {}
        self._model_locks = {}
        self._last_access = {}
        self._model_metadata = {}
        self._lock = threading.RLock()
        
        # Configuration
        self.max_models_in_memory = int(os.environ.get('MAX_MODELS_IN_MEMORY', '3'))
        self.model_ttl_hours = int(os.environ.get('MODEL_TTL_HOURS', '2'))
        
    def get_model(self, model_name: str) -> Optional[Any]:
        """
        Get model with lazy loading and memory management
        
        Args:
            model_name: Name of the model ('crop_recommendation', 'yield_prediction')
            
        Returns:
            Loaded model or None if not available
        """
        with self._lock:
            # Check if model is already loaded and not expired
            if model_name in self._models:
                last_access = self._last_access.get(model_name)
                if last_access and (datetime.now() - last_access).total_seconds() < self.model_ttl_hours * 3600:
                    self._last_access[model_name] = datetime.now()
                    return self._models[model_name]
                else:
                    # Model expired, remove it
                    self._unload_model(model_name)
            
            # Load model if not in memory
            return self._load_model(model_name)
    
    def _load_model(self, model_name: str) -> Optional[Any]:
        """Load model from disk with memory management"""
        try:
            # Get model-specific lock
            if model_name not in self._model_locks:
                self._model_locks[model_name] = threading.Lock()
            
            with self._model_locks[model_name]:
                # Double-check if model was loaded by another thread
                if model_name in self._models:
                    self._last_access[model_name] = datetime.now()
                    return self._models[model_name]
                
                # Enforce memory limits
                self._enforce_memory_limits()
                
                # Load model based on type
                model_path = self._get_model_path(model_name)
                if not os.path.exists(model_path):
                    logger.warning(f"Model {model_name} not found at {model_path}")
                    return None
                
                logger.info(f"Loading model: {model_name}")
                model = load(model_path)
                
                # Store model and metadata
                self._models[model_name] = model
                self._last_access[model_name] = datetime.now()
                self._model_metadata[model_name] = {
                    'loaded_at': datetime.now(),
                    'file_path': model_path,
                    'file_size': os.path.getsize(model_path)
                }
                
                logger.info(f"Model {model_name} loaded successfully")
                return model
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def _enforce_memory_limits(self):
        """Remove least recently used models if memory limit exceeded"""
        if len(self._models) >= self.max_models_in_memory:
            # Find least recently used model
            lru_model = min(self._last_access.items(), key=lambda x: x[1])
            lru_model_name = lru_model[0]
            
            logger.info(f"Memory limit reached. Unloading LRU model: {lru_model_name}")
            self._unload_model(lru_model_name)
    
    def _unload_model(self, model_name: str):
        """Unload model from memory"""
        try:
            if model_name in self._models:
                del self._models[model_name]
            if model_name in self._last_access:
                del self._last_access[model_name]
            if model_name in self._model_metadata:
                del self._model_metadata[model_name]
            
            logger.info(f"Model {model_name} unloaded from memory")
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {str(e)}")
    
    def _get_model_path(self, model_name: str) -> str:
        """Get file path for model"""
        model_paths = {
            'crop_recommendation': 'static/models/crop_recommendation_model.joblib',
            'yield_prediction': 'static/models/yield_prediction_model.joblib'
        }
        return model_paths.get(model_name, f'static/models/{model_name}_model.joblib')
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        with self._lock:
            status = {
                'loaded_models': list(self._models.keys()),
                'memory_usage': {
                    'models_in_memory': len(self._models),
                    'max_models': self.max_models_in_memory,
                    'model_ttl_hours': self.model_ttl_hours
                },
                'model_details': {}
            }
            
            for model_name, metadata in self._model_metadata.items():
                status['model_details'][model_name] = {
                    'loaded_at': metadata['loaded_at'].isoformat(),
                    'file_size_bytes': metadata['file_size'],
                    'last_access': self._last_access[model_name].isoformat()
                }
            
            return status
    
    def clear_all_models(self):
        """Clear all models from memory"""
        with self._lock:
            model_names = list(self._models.keys())
            for model_name in model_names:
                self._unload_model(model_name)
            logger.info("All models cleared from memory")
    
    def preload_model(self, model_name: str) -> bool:
        """Preload a specific model"""
        try:
            model = self.get_model(model_name)
            return model is not None
        except Exception as e:
            logger.error(f"Error preloading model {model_name}: {str(e)}")
            return False

class LabelEncoderRegistry:
    """
    Registry for label encoders with lazy loading
    """
    
    def __init__(self):
        self._encoders = {}
        self._encoder_locks = {}
        self._lock = threading.RLock()
    
    def get_encoder(self, encoder_name: str) -> Optional[Any]:
        """Get label encoder with lazy loading"""
        with self._lock:
            if encoder_name not in self._encoders:
                self._load_encoder(encoder_name)
            
            return self._encoders.get(encoder_name)
    
    def _load_encoder(self, encoder_name: str):
        """Load encoder from disk"""
        try:
            # Get encoder-specific lock
            if encoder_name not in self._encoder_locks:
                self._encoder_locks[encoder_name] = threading.Lock()
            
            with self._encoder_locks[encoder_name]:
                # Double-check if encoder was loaded by another thread
                if encoder_name in self._encoders:
                    return
                
                encoder_path = f'static/labelencoder/{encoder_name}_le.joblib'
                if not os.path.exists(encoder_path):
                    logger.warning(f"Label encoder {encoder_name} not found at {encoder_path}")
                    return
                
                logger.info(f"Loading label encoder: {encoder_name}")
                encoder = load(encoder_path)
                self._encoders[encoder_name] = encoder
                
        except Exception as e:
            logger.error(f"Error loading encoder {encoder_name}: {str(e)}")
    
    def get_all_encoders(self) -> Dict[str, Any]:
        """Get all required encoders for yield prediction"""
        encoder_names = ['statename', 'districtname', 'season', 'crop']
        encoders = {}
        
        for name in encoder_names:
            encoder = self.get_encoder(name)
            if encoder is not None:
                encoders[name] = encoder
        
        return encoders
    
    def encode_safe(self, value: str, encoder_name: str, default_index: int = 0) -> int:
        """Safely encode value with fallback for unknown values"""
        encoder = self.get_encoder(encoder_name)
        if encoder is None:
            return default_index
        
        try:
            return int(encoder.transform([value.lower()])[0])
        except ValueError:
            # Use hash-based encoding for unknown values
            return abs(hash(value.lower())) % len(encoder.classes_)

class DatasetRegistry:
    """
    Registry for dataset management with lazy loading and memory caps
    """
    
    def __init__(self):
        self._datasets = {}
        self._dataset_locks = {}
        self._last_access = {}
        self._lock = threading.RLock()
        
        # Configuration
        self.max_datasets_in_memory = int(os.environ.get('MAX_DATASETS_IN_MEMORY', '2'))
        self.dataset_ttl_hours = int(os.environ.get('DATASET_TTL_HOURS', '1'))
    
    def get_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Get dataset with lazy loading and memory management"""
        with self._lock:
            # Check if dataset is already loaded and not expired
            if dataset_name in self._datasets:
                last_access = self._last_access.get(dataset_name)
                if last_access and (datetime.now() - last_access).total_seconds() < self.dataset_ttl_hours * 3600:
                    self._last_access[dataset_name] = datetime.now()
                    return self._datasets[dataset_name]
                else:
                    # Dataset expired, remove it
                    self._unload_dataset(dataset_name)
            
            # Load dataset if not in memory
            return self._load_dataset(dataset_name)
    
    def _load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load dataset from disk with memory management"""
        try:
            # Get dataset-specific lock
            if dataset_name not in self._dataset_locks:
                self._dataset_locks[dataset_name] = threading.Lock()
            
            with self._dataset_locks[dataset_name]:
                # Double-check if dataset was loaded by another thread
                if dataset_name in self._datasets:
                    self._last_access[dataset_name] = datetime.now()
                    return self._datasets[dataset_name]
                
                # Enforce memory limits
                self._enforce_memory_limits()
                
                # Load dataset based on type
                dataset_path = self._get_dataset_path(dataset_name)
                if not os.path.exists(dataset_path):
                    logger.warning(f"Dataset {dataset_name} not found at {dataset_path}")
                    return None
                
                logger.info(f"Loading dataset: {dataset_name}")
                
                # Use chunked reading for large files
                file_size = os.path.getsize(dataset_path)
                if file_size > 10 * 1024 * 1024:  # 10MB
                    # Load in chunks for large files
                    df = pd.read_csv(dataset_path, chunksize=1000, low_memory=True)
                    df = pd.concat(df, ignore_index=True)
                else:
                    df = pd.read_csv(dataset_path, low_memory=True)
                
                # Store dataset and metadata
                self._datasets[dataset_name] = df
                self._last_access[dataset_name] = datetime.now()
                
                logger.info(f"Dataset {dataset_name} loaded with {len(df)} records")
                return df
                
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            return None
    
    def _enforce_memory_limits(self):
        """Remove least recently used datasets if memory limit exceeded"""
        if len(self._datasets) >= self.max_datasets_in_memory:
            # Find least recently used dataset
            lru_dataset = min(self._last_access.items(), key=lambda x: x[1])
            lru_dataset_name = lru_dataset[0]
            
            logger.info(f"Memory limit reached. Unloading LRU dataset: {lru_dataset_name}")
            self._unload_dataset(lru_dataset_name)
    
    def _unload_dataset(self, dataset_name: str):
        """Unload dataset from memory"""
        try:
            if dataset_name in self._datasets:
                del self._datasets[dataset_name]
            if dataset_name in self._last_access:
                del self._last_access[dataset_name]
            
            logger.info(f"Dataset {dataset_name} unloaded from memory")
        except Exception as e:
            logger.error(f"Error unloading dataset {dataset_name}: {str(e)}")
    
    def _get_dataset_path(self, dataset_name: str) -> str:
        """Get file path for dataset"""
        dataset_paths = {
            'crop_recommendation': 'static/datasets/crop_recommendation.csv',
            'crop_yield_indian_states': 'static/datasets/crop_yield_indian_states.csv',
            'yield_prediction': 'static/datasets/yield_prediction.csv',
            'market_prices': 'static/datasets/market_prices.csv',
            'disease_prediction': 'static/datasets/disease_prediction.csv',
            'fertilizer_recommendation': 'static/datasets/fertilizer_recommendation.csv'
        }
        return dataset_paths.get(dataset_name, f'static/datasets/{dataset_name}.csv')
    
    def get_dataset_status(self) -> Dict[str, Any]:
        """Get status of all datasets"""
        with self._lock:
            status = {
                'loaded_datasets': list(self._datasets.keys()),
                'memory_usage': {
                    'datasets_in_memory': len(self._datasets),
                    'max_datasets': self.max_datasets_in_memory,
                    'dataset_ttl_hours': self.dataset_ttl_hours
                },
                'available_datasets': []
            }
            
            # Check available datasets on disk
            datasets_dir = 'static/datasets'
            if os.path.exists(datasets_dir):
                for file in os.listdir(datasets_dir):
                    if file.endswith('.csv'):
                        dataset_name = file.replace('.csv', '')
                        file_path = os.path.join(datasets_dir, file)
                        file_size = os.path.getsize(file_path)
                        
                        status['available_datasets'].append({
                            'name': dataset_name,
                            'file_path': file_path,
                            'file_size_bytes': file_size,
                            'loaded': dataset_name in self._datasets
                        })
            
            return status

class PerformanceMonitor:
    """
    Monitor application performance and resource usage
    """
    
    def __init__(self):
        self._start_time = datetime.now()
        self._request_count = 0
        self._error_count = 0
        self._response_times = []
        self._lock = threading.Lock()
    
    def record_request(self, endpoint: str, response_time: float, success: bool = True):
        """Record request metrics"""
        with self._lock:
            self._request_count += 1
            if not success:
                self._error_count += 1
            
            self._response_times.append(response_time)
            
            # Keep only last 1000 response times to prevent memory growth
            if len(self._response_times) > 1000:
                self._response_times = self._response_times[-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        with self._lock:
            uptime = (datetime.now() - self._start_time).total_seconds()
            
            if self._response_times:
                avg_response_time = sum(self._response_times) / len(self._response_times)
                p95_response_time = np.percentile(self._response_times, 95)
            else:
                avg_response_time = 0
                p95_response_time = 0
            
            return {
                'uptime_seconds': uptime,
                'total_requests': self._request_count,
                'error_count': self._error_count,
                'error_rate': self._error_count / max(self._request_count, 1),
                'avg_response_time_ms': avg_response_time * 1000,
                'p95_response_time_ms': p95_response_time * 1000,
                'requests_per_second': self._request_count / max(uptime, 1)
            }

# Global registries
model_registry = ModelRegistry()
encoder_registry = LabelEncoderRegistry()
dataset_registry = DatasetRegistry()
performance_monitor = PerformanceMonitor()

def get_crop_recommendation_model():
    """Get crop recommendation model with lazy loading"""
    return model_registry.get_model('crop_recommendation')

def get_yield_prediction_model():
    """Get yield prediction model with lazy loading"""
    return model_registry.get_model('yield_prediction')

def get_label_encoders():
    """Get all label encoders for yield prediction"""
    return encoder_registry.get_all_encoders()

def get_dataset(dataset_name: str):
    """Get dataset with lazy loading"""
    return dataset_registry.get_dataset(dataset_name)

def get_system_status():
    """Get comprehensive system status"""
    return {
        'models': model_registry.get_model_status(),
        'datasets': dataset_registry.get_dataset_status(),
        'performance': performance_monitor.get_metrics(),
        'timestamp': datetime.now().isoformat()
    }

def cleanup_resources():
    """Clean up all resources"""
    logger.info("Cleaning up all resources...")
    model_registry.clear_all_models()
    # dataset_registry doesn't need explicit cleanup as it uses TTL
    logger.info("Resource cleanup completed")
