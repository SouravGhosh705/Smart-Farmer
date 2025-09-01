#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Production server for Smart Farmer API using Waitress
This eliminates the Flask development server warning
"""

from waitress import serve
from app import app
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("🚀 Starting Smart Farmer API with Waitress Production Server")
    logger.info("🌾 Backend will be available at: http://localhost:8000")
    logger.info("📱 Frontend should be at: http://localhost:3000")
    logger.info("🔥 No more development server warnings!")
    
    # Serve the Flask app with Waitress (production-ready server)
    serve(
        app, 
        host='0.0.0.0', 
        port=8000,
        threads=6,  # Handle multiple concurrent requests
        cleanup_interval=30,  # Clean up connections every 30 seconds
        connection_limit=1000,  # Maximum connections
        channel_timeout=120  # Request timeout
    )
