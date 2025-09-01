from flask import Flask, jsonify, redirect
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        'message': 'SmartFarmer REST API - Redirected to Main Backend',
        'main_backend': 'http://localhost:8000',
        'status': 'Use main backend for all API calls'
    })

@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def redirect_to_main_backend(path):
    """Redirect all API calls to main backend"""
    return jsonify({
        'redirect': f'http://localhost:8000/{path}',
        'message': 'Please use the main backend at localhost:8000 for all API endpoints'
    }), 302

if __name__ == '__main__':
    app.run(host='localhost', port=8001, debug=True)  # Different port to avoid conflict
