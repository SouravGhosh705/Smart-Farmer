import React, { useState, useEffect } from 'react';
import { useLanguage } from './contexts/LanguageContext';

const LanguageSelector = () => {
  const { currentLanguage, changeLanguage, getAvailableLanguages, translateText, isLoading } = useLanguage();
  const [label, setLabel] = useState('Language');
  const [loadingText, setLoadingText] = useState('Loading languages...');

  // Get translated labels
  useEffect(() => {
    const loadTranslations = async () => {
      const translatedLabel = await translateText('language_selector', 'Language');
      const translatedLoading = await translateText('loading', 'Loading...');
      setLabel(translatedLabel);
      setLoadingText(translatedLoading);
    };
    loadTranslations();
  }, [currentLanguage, translateText]);

  const handleLanguageChange = (event) => {
    const selectedLanguage = event.target.value;
    changeLanguage(selectedLanguage);
  };

  if (isLoading) {
    return (
      <div className="language-selector" style={{ 
        position: 'fixed',
        top: '10px',
        right: '10px',
        zIndex: 1000,
        backgroundColor: 'white',
        padding: '8px 12px',
        borderRadius: '8px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.15)'
      }}>
        <select disabled style={{
          border: '1px solid #ddd',
          borderRadius: '4px',
          padding: '5px 10px',
          fontSize: '14px'
        }}>
          <option>{loadingText}</option>
        </select>
      </div>
    );
  }

  return (
    <div className="language-selector" style={{ 
      position: 'fixed',
      top: '10px',
      right: '10px',
      zIndex: 1000,
      backgroundColor: 'white',
      padding: '8px 12px',
      borderRadius: '8px',
      boxShadow: '0 2px 10px rgba(0,0,0,0.15)',
      border: '1px solid #e0e0e0',
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    }}>
      <span style={{ 
        fontWeight: '500', 
        color: '#2e7d32',
        fontSize: '14px'
      }}>🌍 {label}:</span>
      <select 
        value={currentLanguage} 
        onChange={handleLanguageChange}
        style={{
          border: '1px solid #ddd',
          borderRadius: '4px',
          padding: '5px 10px',
          fontSize: '14px',
          backgroundColor: 'white',
          cursor: 'pointer',
          minWidth: '120px'
        }}
      >
        {getAvailableLanguages().map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.native}
          </option>
        ))}
      </select>
    </div>
  );
};

export default LanguageSelector;
