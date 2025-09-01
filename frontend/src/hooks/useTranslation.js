import { useState, useEffect } from 'react';
import { useLanguage } from '../contexts/LanguageContext';

// Hook for simple text translation
export const useTranslation = (key, fallback = key) => {
    const { translateText, currentLanguage } = useLanguage();
    const [translation, setTranslation] = useState(fallback);

    useEffect(() => {
        const getTranslation = async () => {
            const translated = await translateText(key, fallback);
            setTranslation(translated);
        };

        getTranslation();
    }, [key, fallback, currentLanguage, translateText]);

    return translation;
};

// Hook for batch translations
export const useBatchTranslation = (textKeys) => {
    const { translateTexts, currentLanguage } = useLanguage();
    const [translations, setTranslations] = useState(textKeys); // Initialize with fallbacks
    const [loading, setLoading] = useState(false); // Start as false, only load if not English

    useEffect(() => {
        // Skip translation for English
        if (currentLanguage === 'en') {
            setTranslations(textKeys);
            setLoading(false);
            return;
        }

        const loadTranslations = async () => {
            // Set a very short loading state to prevent UI blocking
            setLoading(true);
            
            // Use immediate timeout to prevent hanging
            const timeoutId = setTimeout(() => {
                console.warn('Translation taking too long, using fallbacks');
                setTranslations(textKeys);
                setLoading(false);
            }, 1000); // Only 1 second timeout
            
            try {
                const result = await translateTexts(textKeys);
                clearTimeout(timeoutId);
                setTranslations(result);
                setLoading(false);
            } catch (error) {
                clearTimeout(timeoutId);
                console.error('Batch translation error:', error);
                // Fallback to original keys
                setTranslations(textKeys);
                setLoading(false);
            }
        };

        loadTranslations();
    }, [currentLanguage, translateTexts, textKeys]);

    return { translations, loading };
};

// Hook for translating API responses
export const useApiTranslation = () => {
    const { getApiLanguage, currentLanguage } = useLanguage();

    const translateApiRequest = (requestData) => {
        return {
            ...requestData,
            language: getApiLanguage()
        };
    };

    const translateCropName = async (cropName) => {
        if (currentLanguage === 'en') return cropName;

        try {
            const response = await fetch('http://localhost:8000/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    type: 'crop',
                    text: cropName,
                    language: getApiLanguage()
                })
            });

            if (response.ok) {
                const data = await response.json();
                return data.translated || cropName;
            }
        } catch (error) {
            console.warn('Crop translation error:', error);
        }

        return cropName;
    };

    return {
        translateApiRequest,
        translateCropName,
        getApiLanguage
    };
};

export default useTranslation;
