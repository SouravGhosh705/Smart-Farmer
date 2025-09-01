#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ollama Setup and Configuration for Smart Chatbot
Handles Ollama installation, model management, and agricultural knowledge integration
"""

import os
import subprocess
import json
import logging
import requests
import time
from typing import Dict, List, Optional
import asyncio
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)

class OllamaManager:
    """Manager for Ollama installation and model management"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.models_dir = "static/ollama_models"
        self.config_file = "static/ollama_config.json"
        
        # Recommended models for agricultural applications
        self.recommended_models = {
            'llama3.2': {
                'name': 'Llama 3.2',
                'size': '4.7GB',
                'description': 'Latest Llama model, good for general agricultural advice',
                'best_for': ['general farming advice', 'crop recommendations', 'pest management'],
                'min_ram': '8GB'
            },
            'llama3.1': {
                'name': 'Llama 3.1', 
                'size': '4.7GB',
                'description': 'Stable Llama model with good agricultural knowledge',
                'best_for': ['detailed farming explanations', 'technical advice'],
                'min_ram': '8GB'
            },
            'mistral': {
                'name': 'Mistral 7B',
                'size': '4.1GB', 
                'description': 'Efficient model for quick responses',
                'best_for': ['quick answers', 'basic farming queries'],
                'min_ram': '6GB'
            },
            'codellama': {
                'name': 'Code Llama',
                'size': '3.8GB',
                'description': 'Good for technical agricultural calculations',
                'best_for': ['fertilizer calculations', 'yield analysis'],
                'min_ram': '6GB'
            },
            'phi3': {
                'name': 'Phi-3 Mini',
                'size': '2.3GB',
                'description': 'Lightweight model for basic responses',
                'best_for': ['simple queries', 'resource-constrained systems'],
                'min_ram': '4GB'
            }
        }
        
        # Agricultural knowledge base for system prompts
        self.agricultural_context = {
            'system_role': 'expert agricultural advisor',
            'expertise_areas': [
                'crop management', 'soil health', 'pest control', 'disease management',
                'fertilizer recommendations', 'irrigation planning', 'harvest timing',
                'post-harvest handling', 'market analysis', 'sustainable farming'
            ],
            'response_guidelines': [
                'Provide practical, actionable advice',
                'Consider local climate and soil conditions',
                'Prioritize sustainable and organic methods when possible',
                'Include safety warnings for chemical applications',
                'Suggest consulting local extension services for complex issues'
            ]
        }
    
    def check_ollama_installation(self) -> Dict:
        """Check if Ollama is installed on the system"""
        try:
            # Try to run ollama version command
            result = subprocess.run(['ollama', '--version'], 
                                 capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                return {
                    'installed': True,
                    'version': version,
                    'status': 'ready'
                }
            else:
                return {
                    'installed': False,
                    'error': result.stderr,
                    'status': 'not_found'
                }
        
        except FileNotFoundError:
            return {
                'installed': False,
                'error': 'Ollama not found in PATH',
                'status': 'not_installed'
            }
        except subprocess.TimeoutExpired:
            return {
                'installed': False,
                'error': 'Ollama command timed out',
                'status': 'timeout'
            }
        except Exception as e:
            return {
                'installed': False,
                'error': str(e),
                'status': 'error'
            }
    
    async def check_ollama_service(self) -> Dict:
        """Check if Ollama service is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        models_data = await response.json()
                        return {
                            'service_running': True,
                            'available_models': [model['name'] for model in models_data.get('models', [])],
                            'status': 'operational'
                        }
                    else:
                        return {
                            'service_running': False,
                            'error': f'Service responded with status {response.status}',
                            'status': 'error'
                        }
        
        except Exception as e:
            return {
                'service_running': False,
                'error': str(e),
                'status': 'not_running'
            }
    
    def get_installation_instructions(self) -> Dict:
        """Get platform-specific installation instructions"""
        import platform
        system = platform.system().lower()
        
        instructions = {
            'windows': {
                'method': 'Download and run installer',
                'steps': [
                    'Go to https://ollama.ai/download',
                    'Download Ollama for Windows',
                    'Run the installer as administrator',
                    'Restart your terminal/command prompt',
                    'Run "ollama --version" to verify installation'
                ],
                'download_url': 'https://ollama.ai/download/windows'
            },
            'linux': {
                'method': 'Command line installation',
                'steps': [
                    'Open terminal',
                    'Run: curl -fsSL https://ollama.ai/install.sh | sh',
                    'Wait for installation to complete',
                    'Run "ollama --version" to verify'
                ],
                'install_command': 'curl -fsSL https://ollama.ai/install.sh | sh'
            },
            'darwin': {  # macOS
                'method': 'Download and install',
                'steps': [
                    'Go to https://ollama.ai/download',
                    'Download Ollama for macOS',
                    'Open the downloaded file and follow instructions',
                    'Run "ollama --version" in terminal to verify'
                ],
                'download_url': 'https://ollama.ai/download/mac'
            }
        }
        
        current_platform = instructions.get(system, instructions['linux'])
        current_platform['detected_platform'] = system
        
        return current_platform
    
    async def pull_model(self, model_name: str) -> Dict:
        """Download/pull a model using Ollama"""
        try:
            if model_name not in self.recommended_models:
                return {
                    'error': f'Model {model_name} not in recommended list',
                    'available_models': list(self.recommended_models.keys())
                }
            
            logger.info(f"Pulling model {model_name}...")
            
            # Use Ollama API to pull model
            payload = {"name": model_name}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 minutes timeout
                ) as response:
                    if response.status == 200:
                        return {
                            'status': 'success',
                            'model': model_name,
                            'info': self.recommended_models[model_name],
                            'message': f'Model {model_name} downloaded successfully'
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'status': 'error',
                            'error': f'Failed to pull model: {response.status}',
                            'details': error_text
                        }
        
        except Exception as e:
            logger.error(f"Model pull error: {str(e)}")
            return {'error': str(e)}
    
    async def list_installed_models(self) -> Dict:
        """List models installed in Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        
                        model_list = []
                        for model in models:
                            model_info = {
                                'name': model['name'],
                                'size': model.get('size', 0),
                                'modified': model.get('modified_at', ''),
                                'digest': model.get('digest', '')
                            }
                            
                            # Add recommendation info if available
                            base_name = model['name'].split(':')[0]
                            if base_name in self.recommended_models:
                                model_info['recommendation'] = self.recommended_models[base_name]
                            
                            model_list.append(model_info)
                        
                        return {
                            'installed_models': model_list,
                            'total_models': len(model_list),
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        return {'error': f'API error: {response.status}'}
        
        except Exception as e:
            return {'error': str(e)}
    
    async def test_model_response(self, model_name: str, test_prompt: str = None) -> Dict:
        """Test a model with an agricultural prompt"""
        try:
            if not test_prompt:
                test_prompt = "What are the best practices for rice cultivation during monsoon season?"
            
            agricultural_system_prompt = f"""You are an expert agricultural advisor specializing in {', '.join(self.agricultural_context['expertise_areas'])}. 
            Provide practical, actionable farming advice based on scientific principles and sustainable practices.
            
            Guidelines:
            - Keep responses concise but comprehensive
            - Prioritize organic and sustainable methods
            - Include safety warnings for chemical applications
            - Consider local climate and soil conditions
            - Suggest consulting extension services for complex issues
            """
            
            full_prompt = f"{agricultural_system_prompt}\n\nFarmer's Question: {test_prompt}\n\nResponse:"
            
            payload = {
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 300
                }
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_time = time.time() - start_time
                        
                        return {
                            'status': 'success',
                            'model': model_name,
                            'prompt': test_prompt,
                            'response': result.get('response', ''),
                            'response_time': round(response_time, 2),
                            'tokens_generated': len(result.get('response', '').split()),
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'status': 'error',
                            'error': f'Model test failed: {response.status}',
                            'details': error_text
                        }
        
        except Exception as e:
            return {'error': str(e)}
    
    def create_agricultural_system_prompt(self, context: Dict = None) -> str:
        """Create system prompt optimized for agricultural advice"""
        base_prompt = f"""You are an expert agricultural advisor and crop consultant with deep knowledge in:
        {', '.join(self.agricultural_context['expertise_areas'])}.
        
        Your role is to provide helpful, accurate, and practical farming advice to farmers and agricultural professionals.
        
        Guidelines for responses:
        """
        
        for guideline in self.agricultural_context['response_guidelines']:
            base_prompt += f"\n- {guideline}"
        
        if context:
            base_prompt += f"\n\nCurrent Context:\n{json.dumps(context, indent=2)}"
        
        base_prompt += "\n\nAlways provide responses that are:"
        base_prompt += "\n- Practical and implementable"
        base_prompt += "\n- Scientifically accurate"
        base_prompt += "\n- Economically viable for small to medium farmers"
        base_prompt += "\n- Environmentally sustainable"
        base_prompt += "\n- Culturally appropriate for the region"
        
        return base_prompt
    
    def save_configuration(self, config: Dict) -> bool:
        """Save Ollama configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            config['last_updated'] = datetime.now().isoformat()
            config['agricultural_context'] = self.agricultural_context
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def load_configuration(self) -> Dict:
        """Load Ollama configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                return self.get_default_configuration()
        
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return self.get_default_configuration()
    
    def get_default_configuration(self) -> Dict:
        """Get default Ollama configuration"""
        return {
            'base_url': self.base_url,
            'default_model': 'llama3.2',
            'temperature': 0.7,
            'max_tokens': 500,
            'timeout': 30,
            'agricultural_mode': True,
            'multilingual_support': True,
            'safety_filtering': True,
            'response_format': 'markdown'
        }

class AgriculturePromptTemplates:
    """Templates for agricultural prompts and responses"""
    
    def __init__(self):
        self.templates = {
            'crop_recommendation': """
            Based on the following parameters, recommend the best crops:
            Location: {location}
            Soil Type: {soil_type}
            Climate: {climate}
            Season: {season}
            Water Availability: {water_availability}
            
            Please provide:
            1. Top 3 recommended crops with reasons
            2. Expected yield estimates
            3. Key cultivation tips
            4. Potential challenges and solutions
            """,
            
            'disease_diagnosis': """
            Help diagnose and treat this plant health issue:
            Crop: {crop}
            Symptoms: {symptoms}
            Location: {location}
            Weather Conditions: {weather}
            
            Please provide:
            1. Likely disease/pest identification
            2. Organic treatment options
            3. Chemical treatment if necessary
            4. Prevention measures
            5. Expected recovery time
            """,
            
            'general_farming': """
            Provide comprehensive farming advice for:
            Query: {query}
            Crop: {crop}
            Location: {location}
            Experience Level: {farmer_experience}
            
            Please include:
            1. Direct answer to the query
            2. Best practices
            3. Common mistakes to avoid
            4. Cost-effective solutions
            5. Next steps
            """,
            
            'market_analysis': """
            Provide market insights for:
            Crop: {crop}
            Location: {location}
            Current Season: {season}
            
            Please analyze:
            1. Current market trends
            2. Price forecasting
            3. Best selling strategies
            4. Storage recommendations
            5. Alternative marketing channels
            """,
            
            'weather_advisory': """
            Provide weather-based farming advice:
            Current Weather: {weather_description}
            Temperature: {temperature}°C
            Humidity: {humidity}%
            Forecast: {forecast}
            Crop: {crop}
            Growth Stage: {growth_stage}
            
            Please advise on:
            1. Immediate actions needed
            2. Irrigation adjustments
            3. Disease/pest risks
            4. Protection measures
            5. Upcoming planning
            """
        }
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt template with provided parameters"""
        try:
            template = self.templates.get(template_name, self.templates['general_farming'])
            return template.format(**kwargs)
        except Exception as e:
            logger.error(f"Prompt formatting error: {str(e)}")
            return f"Please provide farming advice for: {kwargs.get('query', 'general farming question')}"

class OllamaSetupAssistant:
    """Assistant for setting up Ollama for agricultural use"""
    
    def __init__(self):
        self.manager = OllamaManager()
        self.prompt_templates = AgriculturePromptTemplates()
    
    async def complete_setup(self) -> Dict:
        """Complete Ollama setup process"""
        try:
            setup_results = {
                'steps_completed': [],
                'steps_failed': [],
                'overall_status': 'in_progress'
            }
            
            # Step 1: Check installation
            logger.info("Step 1: Checking Ollama installation...")
            install_check = self.manager.check_ollama_installation()
            
            if install_check['installed']:
                setup_results['steps_completed'].append('Installation verified')
                logger.info("✅ Ollama is installed")
            else:
                setup_results['steps_failed'].append(f"Installation check failed: {install_check['error']}")
                logger.error("❌ Ollama not installed")
                
                instructions = self.manager.get_installation_instructions()
                return {
                    'status': 'installation_required',
                    'instructions': instructions,
                    'setup_results': setup_results
                }
            
            # Step 2: Check service
            logger.info("Step 2: Checking Ollama service...")
            service_check = await self.manager.check_ollama_service()
            
            if service_check['service_running']:
                setup_results['steps_completed'].append('Service is running')
                logger.info("✅ Ollama service is running")
            else:
                setup_results['steps_failed'].append(f"Service check failed: {service_check['error']}")
                logger.warning("⚠️ Ollama service not running")
                
                return {
                    'status': 'service_not_running',
                    'message': 'Please start Ollama service by running "ollama serve" in terminal',
                    'setup_results': setup_results
                }
            
            # Step 3: Check for models
            logger.info("Step 3: Checking installed models...")
            models_check = await self.manager.list_installed_models()
            
            if models_check.get('total_models', 0) > 0:
                setup_results['steps_completed'].append('Models available')
                logger.info(f"✅ Found {models_check['total_models']} installed models")
            else:
                setup_results['steps_completed'].append('Ready for model installation')
                logger.info("ℹ️ No models installed yet")
            
            # Step 4: Save configuration
            logger.info("Step 4: Saving configuration...")
            config = self.manager.get_default_configuration()
            config['setup_completed'] = True
            config['setup_date'] = datetime.now().isoformat()
            
            if self.manager.save_configuration(config):
                setup_results['steps_completed'].append('Configuration saved')
                logger.info("✅ Configuration saved")
            else:
                setup_results['steps_failed'].append('Configuration save failed')
                logger.warning("⚠️ Configuration save failed")
            
            # Determine overall status
            if len(setup_results['steps_failed']) == 0:
                setup_results['overall_status'] = 'success'
            elif len(setup_results['steps_completed']) > len(setup_results['steps_failed']):
                setup_results['overall_status'] = 'partial_success'
            else:
                setup_results['overall_status'] = 'failed'
            
            return {
                'status': setup_results['overall_status'],
                'setup_results': setup_results,
                'configuration': config,
                'next_steps': self._get_next_steps(setup_results, models_check),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Setup error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'setup_results': setup_results
            }
    
    def _get_next_steps(self, setup_results: Dict, models_check: Dict) -> List[str]:
        """Get recommended next steps after setup"""
        next_steps = []
        
        if setup_results['overall_status'] == 'success':
            if models_check.get('total_models', 0) == 0:
                next_steps.extend([
                    "Download a recommended model using: POST /setup/model/install/{model_name}",
                    "Recommended models: llama3.2, mistral, or phi3",
                    "Test the chatbot with: POST /chat"
                ])
            else:
                next_steps.extend([
                    "Setup complete! Ollama is ready for use",
                    "Test the chatbot with agricultural queries",
                    "Explore the FastAPI documentation at /docs"
                ])
        elif setup_results['overall_status'] == 'partial_success':
            next_steps.extend([
                "Complete any failed setup steps",
                "Check Ollama service status",
                "Install a recommended model"
            ])
        else:
            next_steps.extend([
                "Install Ollama following the provided instructions",
                "Start Ollama service",
                "Re-run setup process"
            ])
        
        return next_steps
    
    async def install_recommended_model(self, preference: str = 'balanced') -> Dict:
        """Install a model based on preference"""
        try:
            model_preferences = {
                'fast': 'phi3',           # Lightweight and fast
                'balanced': 'llama3.2',   # Good balance of quality and speed
                'quality': 'llama3.1',    # Best quality responses
                'technical': 'codellama'  # Best for technical calculations
            }
            
            recommended_model = model_preferences.get(preference, 'llama3.2')
            
            logger.info(f"Installing recommended model: {recommended_model}")
            result = await self.manager.pull_model(recommended_model)
            
            if result.get('status') == 'success':
                # Test the model
                test_result = await self.manager.test_model_response(
                    recommended_model,
                    "How can I improve soil health in my wheat field?"
                )
                
                return {
                    'installation': result,
                    'test_result': test_result,
                    'status': 'success',
                    'message': f'Model {recommended_model} installed and tested successfully'
                }
            else:
                return result
        
        except Exception as e:
            return {'error': str(e)}

# Setup functions
async def setup_ollama_for_agriculture() -> Dict:
    """Main setup function for Ollama agricultural integration"""
    try:
        assistant = OllamaSetupAssistant()
        result = await assistant.complete_setup()
        return result
    except Exception as e:
        return {'error': str(e)}

async def install_agricultural_model(model_preference: str = 'balanced') -> Dict:
    """Install and configure agricultural model"""
    try:
        assistant = OllamaSetupAssistant()
        result = await assistant.install_recommended_model(model_preference)
        return result
    except Exception as e:
        return {'error': str(e)}

def get_ollama_status() -> Dict:
    """Get comprehensive Ollama status"""
    try:
        manager = OllamaManager()
        
        # Check installation
        install_status = manager.check_ollama_installation()
        
        # Load configuration
        config = manager.load_configuration()
        
        return {
            'installation': install_status,
            'configuration': config,
            'recommended_models': manager.recommended_models,
            'setup_ready': install_status.get('installed', False),
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        return {'error': str(e)}

# Export main functions
__all__ = [
    'OllamaManager',
    'OllamaSetupAssistant', 
    'AgriculturePromptTemplates',
    'setup_ollama_for_agriculture',
    'install_agricultural_model',
    'get_ollama_status'
]
