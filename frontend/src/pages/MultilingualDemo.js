import React, { useState } from 'react';
import axios from 'axios';
import LanguageSelector from '../LanguageSelector';

const MultilingualDemo = () => {
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [testResult, setTestResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const languageMap = {
    'en': 'english',
    'hi': 'hindi',
    'gu': 'gujarati', 
    'pa': 'punjabi',
    'mr': 'marathi',
    'ta': 'tamil',
    'te': 'telugu',
    'bn': 'bengali'
  };

  const testMultilingualPrediction = async () => {
    setLoading(true);
    try {
      const testData = {
        N: 90,
        P: 42,
        K: 43,
        ph: 6.5,
        rainfall: 200,
        state: "gujarat",
        city: "ahmedabad",
        language: languageMap[currentLanguage]
      };

      const response = await axios.post('http://localhost:8000/crop_prediction', testData);
      setTestResult(response.data);
    } catch (error) {
      console.error('Error testing multilingual prediction:', error);
      setTestResult({ error: error.message });
    }
    setLoading(false);
  };

  return (
    <div style={{ 
      padding: '20px', 
      maxWidth: '800px', 
      margin: '0 auto',
      fontFamily: 'Arial, sans-serif'
    }}>
      <h1 style={{ color: '#2e7d32', textAlign: 'center' }}>
        🌍 Multilingual Smart Farmer Demo
      </h1>
      
      <div style={{ 
        backgroundColor: '#f8f9fa', 
        padding: '20px', 
        borderRadius: '10px',
        margin: '20px 0'
      }}>
        <h3>Select Language / भाषा चुनें</h3>
        <LanguageSelector 
          currentLanguage={currentLanguage}
          onLanguageChange={setCurrentLanguage}
        />
        
        <div style={{ marginTop: '20px' }}>
          <button 
            onClick={testMultilingualPrediction}
            disabled={loading}
            style={{
              backgroundColor: '#4caf50',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '5px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '16px'
            }}
          >
            {loading ? '⏳ Testing...' : '🧪 Test Multilingual Prediction'}
          </button>
        </div>
      </div>

      {testResult && (
        <div style={{ 
          backgroundColor: testResult.error ? '#ffebee' : '#e8f5e8', 
          padding: '20px', 
          borderRadius: '10px',
          border: `2px solid ${testResult.error ? '#f44336' : '#4caf50'}`
        }}>
          <h3>
            {testResult.error ? '❌ Error' : '✅ Results'}
          </h3>
          
          {testResult.error ? (
            <p style={{ color: '#d32f2f' }}>
              Error: {testResult.error}
            </p>
          ) : (
            <div>
              <p><strong>Language:</strong> {testResult.language || 'english'}</p>
              <p><strong>Top Recommended Crop:</strong> 
                <span style={{ 
                  fontSize: '24px', 
                  fontWeight: 'bold', 
                  color: '#2e7d32',
                  marginLeft: '10px'
                }}>
                  {testResult.crop}
                </span>
              </p>
              
              <h4>🌾 Top 5 Crop Recommendations:</h4>
              <ol>
                {testResult.crop_list && testResult.crop_list.slice(0, 5).map((item, index) => (
                  <li key={index} style={{ 
                    fontSize: '18px', 
                    margin: '5px 0',
                    padding: '5px',
                    backgroundColor: index === 0 ? '#c8e6c9' : 'transparent'
                  }}>
                    <strong>{item[0]}</strong>: {(item[1] * 100).toFixed(1)}% confidence
                  </li>
                ))}
              </ol>

              <div style={{ 
                marginTop: '20px', 
                padding: '10px', 
                backgroundColor: '#fff3e0',
                borderRadius: '5px'
              }}>
                <h4>📊 Test Parameters Used:</h4>
                <ul>
                  <li>Nitrogen (N): 90</li>
                  <li>Phosphorus (P): 42</li>
                  <li>Potassium (K): 43</li>
                  <li>pH: 6.5</li>
                  <li>Rainfall: 200mm</li>
                  <li>Location: Ahmedabad, Gujarat</li>
                  <li>Language: {testResult.language}</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      )}

      <div style={{ 
        marginTop: '30px', 
        padding: '20px', 
        backgroundColor: '#e3f2fd',
        borderRadius: '10px'
      }}>
        <h3>🎯 Multilingual Features Available:</h3>
        <ul style={{ fontSize: '16px', lineHeight: '1.6' }}>
          <li>✅ <strong>8 Indian Languages</strong>: English, Hindi, Gujarati, Punjabi, Marathi, Tamil, Telugu, Bengali</li>
          <li>✅ <strong>Crop Names Translation</strong>: 18+ crops in all languages</li>
          <li>✅ <strong>Disease Names</strong>: 15+ diseases translated</li>
          <li>✅ <strong>Fertilizer Recommendations</strong>: In native languages</li>
          <li>✅ <strong>UI Text Translation</strong>: Interface elements</li>
          <li>✅ <strong>Language Auto-Detection</strong>: Detect user's language</li>
        </ul>
        
        <div style={{ 
          marginTop: '15px', 
          padding: '10px', 
          backgroundColor: '#c8e6c9',
          borderRadius: '5px'
        }}>
          <strong>💡 How it works:</strong> The system uses local CSV datasets instead of expensive APIs, 
          providing FREE multilingual support with 100% accuracy for agricultural terms!
        </div>
      </div>
    </div>
  );
};

export default MultilingualDemo;
