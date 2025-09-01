#!/usr/bin/env python3
"""
Smart Farmer Backend Launcher
=============================
Python script to start the FastAPI backend server with proper error handling
"""

import uvicorn
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Start Smart Farmer Backend')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--reload', action='store_true', default=True, help='Enable auto-reload')
    parser.add_argument('--log-level', default='info', help='Log level')
    
    args = parser.parse_args()
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    print('🔥 Smart Farmer Backend - Starting...')
    print('📡 Services initializing...')
    print(f'🌐 Server will be available at: http://localhost:{args.port}')
    print(f'📚 API Documentation: http://localhost:{args.port}/docs')
    print(f'📖 Alternative Docs: http://localhost:{args.port}/redoc')
    print()
    
    try:
        uvicorn.run(
            'fastapi_app:app',
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        print('\n\n🛑 Smart Farmer Backend stopped by user')
        print('👋 Thank you for using Smart Farmer!')
    except Exception as e:
        print(f'\n❌ Error starting server: {e}')
        print('\n🔧 Troubleshooting tips:')
        print('   1. Check if all dependencies are installed')
        print(f'   2. Ensure port {args.port} is not in use')
        print('   3. Run: pip install -r requirements.txt')
        print('   4. Check if fastapi_app.py exists')
        input('\nPress Enter to exit...')
        sys.exit(1)

if __name__ == '__main__':
    main()
