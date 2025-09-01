import React, { useState, useEffect, useRef, useCallback } from 'react';
import './Chatbot.css';

const Chatbot = ({ isOpen, onToggle, userLocation }) => {
    const [sessionId, setSessionId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isTyping, setIsTyping] = useState(false);
    const [isOfflineMode, setIsOfflineMode] = useState(false);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    
    // Offline responses for common farming questions
    const offlineResponses = {
        greetings: [
            "Hello! I'm your Smart Farmer Assistant. How can I help you today?",
            "Hi there! I'm here to help with your farming questions.",
            "Welcome! What would you like to know about farming?"
        ],
        crops: [
            "For crop recommendations, I'd suggest considering your soil type, climate, and local weather conditions. Popular crops include rice, wheat, cotton, and vegetables based on your region.",
            "Crop selection depends on factors like soil pH, rainfall, temperature, and market demand. Would you like specific advice for a particular season?"
        ],
        weather: [
            "Weather is crucial for farming decisions. Check local forecasts for rainfall, temperature, and humidity. Plan irrigation and pest control accordingly.",
            "Monitor weather patterns for planting and harvesting times. Sudden weather changes can affect crop health."
        ],
        fertilizers: [
            "Fertilizer needs depend on soil testing results. Generally, NPK fertilizers work well for most crops. Consider organic options like compost and manure.",
            "Soil testing is essential before applying fertilizers. Over-fertilization can harm crops and the environment."
        ],
        prices: [
            "Market prices fluctuate based on demand, season, and quality. Check local mandis and online platforms for current rates.",
            "To get better prices, consider timing your harvest and exploring different market channels."
        ],
        general: [
            "That's an interesting question! For detailed farming advice, I recommend consulting with local agricultural experts.",
            "I'd be happy to help! Could you be more specific about what farming topic you'd like to discuss?"
        ]
    };
    
    // Function to get offline response based on message content
    const getOfflineResponse = (message) => {
        const msg = message.toLowerCase();
        
        if (msg.includes('hello') || msg.includes('hi') || msg.includes('hey')) {
            return getRandomResponse('greetings');
        } else if (msg.includes('crop') || msg.includes('plant') || msg.includes('grow')) {
            return getRandomResponse('crops');
        } else if (msg.includes('weather') || msg.includes('rain') || msg.includes('temperature')) {
            return getRandomResponse('weather');
        } else if (msg.includes('fertilizer') || msg.includes('nutrient') || msg.includes('soil')) {
            return getRandomResponse('fertilizers');
        } else if (msg.includes('price') || msg.includes('market') || msg.includes('sell')) {
            return getRandomResponse('prices');
        } else {
            return getRandomResponse('general');
        }
    };
    
    const getRandomResponse = (category) => {
        const responses = offlineResponses[category];
        return responses[Math.floor(Math.random() * responses.length)];
    };

    // Auto-scroll to bottom when new messages arrive
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const initializeChatSession = useCallback(async () => {
        try {
            setIsLoading(true);
            
            // Add timeout to API call
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000);
            
            const response = await fetch(`${API_BASE_URL}/chat/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    city: userLocation?.city || '',
                    state: userLocation?.state || '',
                    language: 'english'
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);

            if (response.ok) {
                const data = await response.json();
                setSessionId(data.session_id);
                setIsOfflineMode(false);
                
                // Add welcome message
                setMessages([{
                    id: Date.now(),
                    type: 'bot',
                    content: data.welcome_message.response,
                    suggestions: data.welcome_message.suggestions,
                    timestamp: new Date().toISOString()
                }]);
            } else {
                throw new Error('API response not ok');
            }
        } catch (error) {
            console.log('Backend not available, switching to offline mode:', error.message);
            setIsOfflineMode(true);
            setSessionId('offline-session');
            
            // Initialize with offline welcome message
            setMessages([{
                id: Date.now(),
                type: 'bot',
                content: '🌾 Welcome to Smart Farmer Assistant! \n\nI\'m currently running in offline mode, but I can still help you with basic farming advice and information. How can I assist you today?',
                suggestions: ['Crop recommendations', 'Weather tips', 'Fertilizer advice', 'Market insights'],
                timestamp: new Date().toISOString()
            }]);
        } finally {
            setIsLoading(false);
        }
    }, [userLocation, API_BASE_URL]);

    // Initialize chat session
    useEffect(() => {
        if (isOpen && !sessionId) {
            initializeChatSession();
        }
    }, [isOpen, sessionId, initializeChatSession]);

    // Focus input when chatbot opens
    useEffect(() => {
        if (isOpen && inputRef.current) {
            inputRef.current.focus();
        }
    }, [isOpen]);

    const sendMessage = async (messageText = null) => {
        const message = messageText || inputMessage.trim();
        if (!message || isLoading) return;

        // Add user message to chat
        const userMessage = {
            id: Date.now(),
            type: 'user',
            content: message,
            timestamp: new Date().toISOString()
        };

        setMessages(prev => [...prev, userMessage]);
        setInputMessage('');
        setIsLoading(true);
        setIsTyping(true);

        // Handle offline mode
        if (isOfflineMode) {
            setTimeout(() => {
                const botMessage = {
                    id: Date.now(),
                    type: 'bot',
                    content: getOfflineResponse(message),
                    suggestions: getSuggestionsForMessage(message),
                    timestamp: new Date().toISOString()
                };
                setMessages(prev => [...prev, botMessage]);
                setIsTyping(false);
                setIsLoading(false);
            }, 1000 + Math.random() * 1000); // Random delay for more natural feel
            return;
        }

        try {
            // Add timeout to prevent hanging
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch(`${API_BASE_URL}/chat/message`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: message
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);

            if (response.ok) {
                const data = await response.json();
                
                // Simulate typing delay
                setTimeout(() => {
                    const botMessage = {
                        id: data.message_id || Date.now(),
                        type: 'bot',
                        content: data.response.response,
                        suggestions: data.response.suggestions || [],
                        recommendations: data.response.recommendations || [],
                        locationData: data.response.location_data,
                        intent: data.intent,
                        timestamp: data.timestamp
                    };

                    setMessages(prev => [...prev, botMessage]);
                    setIsTyping(false);
                }, 1000);
            } else {
                throw new Error('Failed to send message');
            }
        } catch (error) {
            console.log('Switching to offline mode due to error:', error.message);
            setIsOfflineMode(true);
            
            // Provide offline response instead of error
            setTimeout(() => {
                const botMessage = {
                    id: Date.now(),
                    type: 'bot',
                    content: getOfflineResponse(message) + "\n\n💡 *Note: I'm currently in offline mode, but I can still provide basic farming guidance!*",
                    suggestions: getSuggestionsForMessage(message),
                    timestamp: new Date().toISOString()
                };
                setMessages(prev => [...prev, botMessage]);
                setIsTyping(false);
            }, 1000);
        } finally {
            setIsLoading(false);
        }
    };
    
    // Get relevant suggestions based on message content
    const getSuggestionsForMessage = (message) => {
        const msg = message.toLowerCase();
        
        if (msg.includes('crop') || msg.includes('plant')) {
            return ['Soil preparation', 'Seed selection', 'Planting season'];
        } else if (msg.includes('weather')) {
            return ['Irrigation tips', 'Pest control', 'Harvest timing'];
        } else if (msg.includes('fertilizer')) {
            return ['Organic fertilizers', 'Soil testing', 'Application timing'];
        } else if (msg.includes('price')) {
            return ['Quality factors', 'Storage tips', 'Market timing'];
        } else {
            return ['Crop advice', 'Weather tips', 'Fertilizer help', 'Price info'];
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const handleSuggestionClick = (suggestion) => {
        sendMessage(suggestion);
    };

    const formatMessage = (content) => {
        // Convert markdown-like formatting to JSX
        return content.split('\n').map((line, index) => {
            // Handle bold text
            line = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            // Handle emojis and bullet points
            
            return (
                <div key={index} dangerouslySetInnerHTML={{ __html: line }} />
            );
        });
    };

    const WeatherAlert = ({ alerts }) => {
        if (!alerts || alerts.length === 0) return null;

        return (
            <div className="weather-alerts">
                {alerts.map((alert, index) => (
                    <div key={index} className={`alert alert-${alert.priority}`}>
                        <span className="alert-icon">⚠️</span>
                        <span className="alert-message">{alert.message}</span>
                    </div>
                ))}
            </div>
        );
    };

    if (!isOpen) {
        return (
            <div className="chatbot-toggle" onClick={onToggle}>
                <div className="chat-icon">
                    <span>🌾</span>
                </div>
                <div className="chat-badge">Chat</div>
            </div>
        );
    }

    return (
        <div className="chatbot-container">
            <div className="chatbot-header">
                <div className="header-content">
                    <div className="bot-avatar">🌾</div>
                    <div className="bot-info">
                        <h3>Smart Farmer Assistant</h3>
                        <span className="status">
                            {isTyping ? 'Typing...' : (isOfflineMode ? 'Offline Mode' : 'Online')}
                        </span>
                    </div>
                </div>
                <button className="close-btn" onClick={onToggle}>
                    ✕
                </button>
            </div>

            <div className="chatbot-messages">
                {messages.map((message) => (
                    <div key={message.id} className={`message ${message.type}`}>
                        <div className="message-content">
                            <div className="message-text">
                                {formatMessage(message.content)}
                            </div>
                            
                            {message.locationData?.farming_alerts && (
                                <WeatherAlert alerts={message.locationData.farming_alerts} />
                            )}
                            
                            {message.recommendations && message.recommendations.length > 0 && (
                                <div className="recommendations">
                                    <h4>Recommended Crops:</h4>
                                    <div className="crop-chips">
                                        {message.recommendations.map((crop, index) => (
                                            <span key={index} className="crop-chip">
                                                {crop}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                            
                            {message.suggestions && message.suggestions.length > 0 && (
                                <div className="suggestions">
                                    {message.suggestions.map((suggestion, index) => (
                                        <button
                                            key={index}
                                            className="suggestion-btn"
                                            onClick={() => handleSuggestionClick(suggestion)}
                                        >
                                            {suggestion}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                        <div className="message-time">
                            {new Date(message.timestamp).toLocaleTimeString([], { 
                                hour: '2-digit', 
                                minute: '2-digit' 
                            })}
                        </div>
                    </div>
                ))}
                
                {isTyping && (
                    <div className="message bot">
                        <div className="message-content">
                            <div className="typing-indicator">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    </div>
                )}
                
                <div ref={messagesEndRef} />
            </div>

            <div className="chatbot-input">
                <div className="input-container">
                    <textarea
                        ref={inputRef}
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask me about crops, weather, farming tips"
                        disabled={isLoading}
                        rows="1"
                    />
                    <button
                        onClick={() => sendMessage()}
                        disabled={isLoading || !inputMessage.trim()}
                        className="send-btn"
                    >
                        {isLoading ? '⏳' : '📤'}
                    </button>
                </div>
                
                <div className="quick-actions">
                    <button 
                        className="quick-btn"
                        onClick={() => sendMessage('What crops should I plant?')}
                    >
                        🌱 Crop Advice
                    </button>
                    <button 
                        className="quick-btn"
                        onClick={() => sendMessage('Check current weather')}
                    >
                        🌤️ Weather
                    </button>
                    <button 
                        className="quick-btn"
                        onClick={() => sendMessage('Fertilizer recommendations')}
                    >
                        🧪 Fertilizers
                    </button>
                    <button 
                        className="quick-btn"
                        onClick={() => sendMessage('Market prices')}
                    >
                        💰 Prices
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Chatbot;
