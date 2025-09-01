import React, { createContext, useContext, useState, useEffect } from 'react';

// Create Language Context
const LanguageContext = createContext();

// Custom hook to use language context
export const useLanguage = () => {
    const context = useContext(LanguageContext);
    if (!context) {
        throw new Error('useLanguage must be used within a LanguageProvider');
    }
    return context;
};

// Language Provider Component
export const LanguageProvider = ({ children }) => {
    // Get initial language from localStorage or default to English
    const [currentLanguage, setCurrentLanguage] = useState(() => {
        const savedLanguage = localStorage.getItem('smartFarmerLanguage');
        return savedLanguage || 'en';
    });

    // Translation cache to avoid API calls for same translations
    const [translationCache, setTranslationCache] = useState({});

    // Loading state for translations
    const [isLoading, setIsLoading] = useState(false);

    // Supported languages mapping
    const languages = {
        'en': { code: 'en', name: 'English', native: 'English', api: 'english' },
        'hi': { code: 'hi', name: 'Hindi', native: 'हिन्दी', api: 'hindi' },
        'gu': { code: 'gu', name: 'Gujarati', native: 'ગુજરાતી', api: 'gujarati' },
        'pa': { code: 'pa', name: 'Punjabi', native: 'ਪੰਜਾਬੀ', api: 'punjabi' },
        'mr': { code: 'mr', name: 'Marathi', native: 'मराठी', api: 'marathi' },
        'ta': { code: 'ta', name: 'Tamil', native: 'தமிழ்', api: 'tamil' },
        'te': { code: 'te', name: 'Telugu', native: 'తెలుగు', api: 'telugu' },
        'bn': { code: 'bn', name: 'Bengali', native: 'বাংলা', api: 'bengali' }
    };

    // Update localStorage when language changes
    useEffect(() => {
        localStorage.setItem('smartFarmerLanguage', currentLanguage);
    }, [currentLanguage]);

    // Get API language code
    const getApiLanguage = () => {
        return languages[currentLanguage]?.api || 'english';
    };

    // Change language function
    const changeLanguage = (newLanguage) => {
        if (languages[newLanguage]) {
            setCurrentLanguage(newLanguage);
        }
    };

    // Translate UI text function
    const translateText = async (key, fallback = key) => {
        // Return fallback for English
        if (currentLanguage === 'en') {
            return fallback;
        }

        // Check cache first
        const cacheKey = `${currentLanguage}_${key}`;
        if (translationCache[cacheKey]) {
            return translationCache[cacheKey];
        }

        try {
            setIsLoading(true);
            
            // Add very short timeout to prevent hanging
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 2000); // 2 second timeout
            
            const response = await fetch('http://localhost:8000/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    type: 'ui',
                    text: key,
                    language: getApiLanguage()
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);

            if (response.ok) {
                const data = await response.json();
                const translated = data.translated || fallback;
                
                // Update cache
                setTranslationCache(prev => ({
                    ...prev,
                    [cacheKey]: translated
                }));
                
                return translated;
            } else {
                console.warn(`Translation failed for ${key}, using fallback`);
                return fallback;
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                console.warn(`Translation timeout for ${key}, using fallback`);
            } else {
                console.warn(`Translation error for ${key}:`, error);
            }
            return fallback;
        } finally {
            setIsLoading(false);
        }
    };

    // Batch translate multiple UI texts
    const translateTexts = async (texts) => {
        const translations = {};
        
        for (const [key, fallback] of Object.entries(texts)) {
            translations[key] = await translateText(key, fallback);
        }
        
        return translations;
    };

    // Get current language information
    const getCurrentLanguage = () => {
        return languages[currentLanguage] || languages['en'];
    };

    // Get all available languages
    const getAvailableLanguages = () => {
        return Object.values(languages);
    };

    // Check if current language is RTL (none of our supported languages are RTL)
    const isRTL = () => {
        return false; // All our supported languages are LTR
    };

    // Format number according to language locale
    const formatNumber = (number) => {
        const localeMap = {
            'en': 'en-US',
            'hi': 'hi-IN',
            'gu': 'gu-IN',
            'pa': 'pa-IN',
            'mr': 'mr-IN',
            'ta': 'ta-IN',
            'te': 'te-IN',
            'bn': 'bn-IN'
        };

        const locale = localeMap[currentLanguage] || 'en-US';
        return new Intl.NumberFormat(locale).format(number);
    };

    // Format date according to language locale
    const formatDate = (date) => {
        const localeMap = {
            'en': 'en-US',
            'hi': 'hi-IN',
            'gu': 'gu-IN',
            'pa': 'pa-IN',
            'mr': 'mr-IN',
            'ta': 'ta-IN',
            'te': 'te-IN',
            'bn': 'bn-IN'
        };

        const locale = localeMap[currentLanguage] || 'en-US';
        return new Intl.DateTimeFormat(locale).format(new Date(date));
    };

    const value = {
        currentLanguage,
        changeLanguage,
        translateText,
        translateTexts,
        getCurrentLanguage,
        getAvailableLanguages,
        getApiLanguage,
        isRTL,
        formatNumber,
        formatDate,
        isLoading,
        languages
    };

    return (
        <LanguageContext.Provider value={value}>
            {children}
        </LanguageContext.Provider>
    );
};

export default LanguageContext;
