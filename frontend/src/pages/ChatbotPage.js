import React, { useState, useEffect } from 'react';
import Chatbot from '../components/Chatbot';
import './ChatbotPage.css';

const ChatbotPage = () => {
    const [userLocation, setUserLocation] = useState({
        city: '',
        state: ''
    });
    const [locationPermission, setLocationPermission] = useState(false);

    useEffect(() => {
        // Try to get user's location
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    // In a real app, you'd use reverse geocoding to get city/state
                    setLocationPermission(true);
                    // For demo purposes, you can set default location
                    setUserLocation({
                        city: 'Delhi',
                        state: 'Delhi'
                    });
                },
                (error) => {
                    console.log('Location access denied:', error);
                    setLocationPermission(false);
                }
            );
        }
    }, []);

    return (
        <div className="chatbot-page">
            <div className="chatbot-page-header">
                <h1>🌾 Smart Farmer Assistant</h1>
                <p>Get real-time, location-specific crop advisory and farming guidance</p>
            </div>

            <div className="chatbot-features">
                <div className="feature-grid">
                    <div className="feature-card">
                        <div className="feature-icon">🌱</div>
                        <h3>Crop Recommendations</h3>
                        <p>Get personalized crop suggestions based on your location, soil, and weather conditions</p>
                    </div>
                    
                    <div className="feature-card">
                        <div className="feature-icon">🌤️</div>
                        <h3>Weather Advisory</h3>
                        <p>Real-time weather updates and farming alerts for your specific location</p>
                    </div>
                    
                    <div className="feature-card">
                        <div className="feature-icon">🧪</div>
                        <h3>Fertilizer Guidance</h3>
                        <p>Expert advice on nutrient management and organic farming practices</p>
                    </div>
                    
                    <div className="feature-card">
                        <div className="feature-icon">🦠</div>
                        <h3>Disease Management</h3>
                        <p>Early detection tips and treatment recommendations for common crop diseases</p>
                    </div>
                    
                    <div className="feature-card">
                        <div className="feature-icon">📊</div>
                        <h3>Yield Predictions</h3>
                        <p>AI-powered yield forecasting to help you plan and optimize production</p>
                    </div>
                    
                    <div className="feature-card">
                        <div className="feature-icon">💰</div>
                        <h3>Market Insights</h3>
                        <p>Current market prices and trends to help you make informed selling decisions</p>
                    </div>
                </div>
            </div>

            {!locationPermission && (
                <div className="location-prompt">
                    <div className="location-card">
                        <h3>📍 Set Your Location</h3>
                        <p>For personalized farming advice, please provide your location:</p>
                        <div className="location-inputs">
                            <input
                                type="text"
                                placeholder="Enter your city"
                                value={userLocation.city}
                                onChange={(e) => setUserLocation(prev => ({ ...prev, city: e.target.value }))}
                            />
                            <input
                                type="text"
                                placeholder="Enter your state"
                                value={userLocation.state}
                                onChange={(e) => setUserLocation(prev => ({ ...prev, state: e.target.value }))}
                            />
                            <button 
                                onClick={() => setLocationPermission(true)}
                                disabled={!userLocation.city}
                                className="location-btn"
                            >
                                Set Location
                            </button>
                        </div>
                    </div>
                </div>
            )}

            <div className="chat-instructions">
                <h2>🗣️ How to Chat</h2>
                <div className="instruction-grid">
                    <div className="instruction-item">
                        <span className="step-number">1</span>
                        <p>Click the chat button in the bottom-right corner</p>
                    </div>
                    <div className="instruction-item">
                        <span className="step-number">2</span>
                        <p>Ask your farming questions in natural language</p>
                    </div>
                    <div className="instruction-item">
                        <span className="step-number">3</span>
                        <p>Get instant, location-specific advice and recommendations</p>
                    </div>
                </div>
            </div>

            <div className="sample-queries">
                <h3>💬 Try These Sample Questions:</h3>
                <div className="query-examples">
                    <div className="query-item">"What crops should I plant in Delhi this season?"</div>
                    <div className="query-item">"How to manage rice diseases in monsoon?"</div>
                    <div className="query-item">"What fertilizer should I use for wheat?"</div>
                    <div className="query-item">"Current market price of cotton?"</div>
                    <div className="query-item">"When to plant tomatoes in Maharashtra?"</div>
                    <div className="query-item">"How does weather affect crop yield?"</div>
                </div>
            </div>

            {/* The actual chatbot component */}
            <Chatbot 
                isOpen={true} 
                onToggle={() => {}} 
                userLocation={userLocation}
            />
        </div>
    );
};

export default ChatbotPage;
