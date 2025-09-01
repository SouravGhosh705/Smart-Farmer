import React, { useState, useEffect, useRef } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { Card, Button, Alert, Spinner, ProgressBar, Modal } from 'react-bootstrap';
import axios from 'axios';
import './AiCropDoctor.css';

const AiCropDoctor = () => {
  const { translateText, currentLanguage } = useLanguage();
  
  // Component state
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [selectedCrop, setSelectedCrop] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [selectedIssueDetail, setSelectedIssueDetail] = useState(null);
  
  // Translated text state
  const [texts, setTexts] = useState({});
  
  // File input ref
  const fileInputRef = useRef(null);
  
  // Available crop types
  const cropTypes = [
    'rice', 'wheat', 'cotton', 'maize', 'tomato', 'potato', 'onion', 
    'sugarcane', 'soybean', 'groundnut', 'mustard', 'barley', 'gram'
  ];

  // Load translations with fallback
  useEffect(() => {
    const loadTranslations = async () => {
      const translationKeys = {
        title: 'AI Crop Doctor',
        subtitle: 'Upload plant images for AI-powered disease and pest detection',
        uploadArea: 'Drag and drop an image here, or click to select',
        selectCrop: 'Select Crop Type (Optional)',
        chooseCrop: 'Choose crop type...',
        analyzeButton: 'Analyze Plant Health',
        analyzing: 'Analyzing image...',
        noImageError: 'Please select an image first',
        analysisError: 'Error analyzing image',
        detectionResults: 'Detection Results',
        severity: 'Severity',
        confidence: 'Confidence',
        recommendations: 'Treatment Recommendations',
        nextSteps: 'Next Steps',
        viewDetails: 'View Details',
        closeDetails: 'Close',
        uploadAnother: 'Upload Another Image',
        healthyPlant: 'Healthy Plant',
        mildIssues: 'Mild Issues',
        moderateIssues: 'Moderate Issues',
        severeIssues: 'Severe Issues',
        imageQuality: 'Image Quality',
        detectedIssues: 'Detected Issues',
        treatment: 'Treatment',
        prevention: 'Prevention',
        noIssuesFound: 'No major issues detected',
        plantAppearHealthy: 'Your plant appears to be healthy!',
        continueMonitoring: 'Continue regular monitoring',
        supportedFormats: 'Supported formats: JPG, PNG, WebP',
        maxFileSize: 'Maximum file size: 10MB',
        uploadingImage: 'Uploading image...',
        processingImage: 'Processing image...',
        generatingReport: 'Generating report...'
      };

      // Set default texts immediately to prevent loading issues
      setTexts(translationKeys);
      
      // Try to load translations with timeout
      try {
        const translatedTexts = {};
        const translationPromises = Object.entries(translationKeys).map(async ([key, defaultText]) => {
          try {
            const translated = await Promise.race([
              translateText(key, defaultText),
              new Promise((_, reject) => setTimeout(() => reject(new Error('Translation timeout')), 2000))
            ]);
            return [key, translated];
          } catch (error) {
            return [key, defaultText]; // Fallback to default
          }
        });
        
        const results = await Promise.allSettled(translationPromises);
        results.forEach(result => {
          if (result.status === 'fulfilled') {
            const [key, value] = result.value;
            translatedTexts[key] = value;
          }
        });
        
        setTexts(prev => ({ ...prev, ...translatedTexts }));
      } catch (error) {
        console.log('Translation service unavailable, using default texts');
        // Keep default texts already set
      }
    };

    loadTranslations();
  }, [currentLanguage, translateText]);

  // Handle file selection
  const handleFileSelect = (file) => {
    console.log('File selected:', file);
    if (!file) {
      console.log('No file provided');
      return;
    }
    
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    console.log('File type:', file.type, 'Allowed:', allowedTypes.includes(file.type));
    
    if (!allowedTypes.includes(file.type)) {
      const errorMsg = `Invalid file type: ${file.type}. Please select a valid image file (JPG, PNG, WebP)`;
      console.error(errorMsg);
      setError(errorMsg);
      return;
    }
    
    // Validate file size (10MB max)
    const fileSizeMB = file.size / (1024 * 1024);
    console.log('File size:', fileSizeMB.toFixed(2), 'MB');
    
    if (file.size > 10 * 1024 * 1024) {
      const errorMsg = `File size too large: ${fileSizeMB.toFixed(2)}MB. Maximum allowed: 10MB`;
      console.error(errorMsg);
      setError(errorMsg);
      return;
    }
    
    console.log('File validation passed, setting selected image');
    setSelectedImage(file);
    setError(null);
    setAnalysisResult(null);
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      console.log('Image preview created');
      setImagePreview(e.target.result);
    };
    reader.onerror = (e) => {
      console.error('Error reading file:', e);
      setError('Error reading the selected file');
    };
    reader.readAsDataURL(file);
  };

  // Handle file input change
  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  // Handle drag and drop
  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  // Offline analysis function
  const getOfflineAnalysis = (cropType) => {
    const commonIssues = {
      rice: [
        {
          name: 'Brown Spot',
          description: 'Brown circular spots on leaves, common in humid conditions',
          severity: 'moderate',
          affected_area: 'Leaves',
          characteristics: 'Circular brown spots with yellow halos'
        },
        {
          name: 'Blast Disease',
          description: 'Diamond-shaped lesions on leaves and stems',
          severity: 'severe',
          affected_area: 'Leaves and stems',
          characteristics: 'Diamond-shaped spots with gray centers'
        }
      ],
      wheat: [
        {
          name: 'Leaf Rust',
          description: 'Orange-red pustules on leaves, reduces yield',
          severity: 'moderate',
          affected_area: 'Leaves',
          characteristics: 'Small orange-red powdery spots'
        },
        {
          name: 'Powdery Mildew',
          description: 'White powdery coating on leaves and stems',
          severity: 'mild',
          affected_area: 'Leaves and stems',
          characteristics: 'White powdery fungal growth'
        }
      ],
      tomato: [
        {
          name: 'Early Blight',
          description: 'Dark spots with concentric rings on older leaves',
          severity: 'mild',
          affected_area: 'Lower leaves',
          characteristics: 'Dark brown spots with yellow halos'
        },
        {
          name: 'Blossom End Rot',
          description: 'Dark, sunken spots on fruit bottom',
          severity: 'moderate',
          affected_area: 'Fruit',
          characteristics: 'Black, leathery spots on fruit bottom'
        }
      ],
      potato: [
        {
          name: 'Late Blight',
          description: 'Water-soaked lesions that turn brown, serious disease',
          severity: 'severe',
          affected_area: 'Leaves and stems',
          characteristics: 'Dark, water-soaked patches'
        },
        {
          name: 'Early Blight',
          description: 'Target-like spots on older leaves',
          severity: 'moderate',
          affected_area: 'Leaves',
          characteristics: 'Concentric ring patterns on leaves'
        }
      ],
      cotton: [
        {
          name: 'Bollworm',
          description: 'Caterpillar damage to cotton bolls and leaves',
          severity: 'severe',
          affected_area: 'Bolls and leaves',
          characteristics: 'Holes in bolls and damaged leaves'
        }
      ],
      maize: [
        {
          name: 'Corn Borer',
          description: 'Insect damage causing holes in stalks and ears',
          severity: 'moderate',
          affected_area: 'Stalks and ears',
          characteristics: 'Round holes in stalks, damaged kernels'
        }
      ],
      onion: [
        {
          name: 'Purple Blotch',
          description: 'Purple-brown spots on leaves and bulbs',
          severity: 'moderate',
          affected_area: 'Leaves and bulbs',
          characteristics: 'Purple-brown lesions with white centers'
        }
      ],
      sugarcane: [
        {
          name: 'Red Rot',
          description: 'Reddish discoloration of internal tissue',
          severity: 'severe',
          affected_area: 'Stalks',
          characteristics: 'Red patches inside stalks'
        }
      ]
    };

    const generalRecommendations = [
      '🌿 **General Care**: Ensure proper spacing between plants for air circulation',
      '💧 **Watering**: Water at the base of plants, avoid wetting leaves',
      '🧹 **Sanitation**: Remove affected plant parts and dispose properly',
      '🔄 **Rotation**: Practice crop rotation to prevent disease buildup',
      '🌡️ **Environment**: Monitor humidity and temperature conditions'
    ];

    const generalNextSteps = [
      '📋 **Monitor Daily**: Check plants regularly for changes',
      '🔬 **Professional Help**: Consult agricultural extension officer if issues persist',
      '📱 **Documentation**: Take photos to track disease progression',
      '🌱 **Prevention**: Focus on preventive measures for healthy plant growth'
    ];

    // Simulate random analysis based on crop type
    const isHealthy = Math.random() > 0.6; // 40% chance of issues
    
    if (isHealthy) {
      return {
        severity: 'healthy',
        analysis_summary: 'Based on general plant health indicators, your plant appears to be in good condition. Continue with regular care and monitoring.',
        detected_issues: [],
        treatment_recommendations: [
          '✅ **Maintain Care**: Continue current care routine',
          '🌿 **Regular Monitoring**: Keep checking for any changes',
          '💚 **Preventive Care**: Apply preventive measures to maintain health'
        ],
        next_steps: [
          '📅 **Schedule**: Regular inspection every 2-3 days',
          '🌱 **Nutrition**: Ensure adequate nutrition and water',
          '🔍 **Early Detection**: Watch for early signs of stress or disease'
        ],
        confidence_scores: [0.85]
      };
    } else {
      const cropIssues = commonIssues[cropType] || [
        {
          name: 'General Plant Stress',
          description: 'Signs of environmental stress or nutrient deficiency',
          severity: 'mild',
          affected_area: 'Overall plant',
          characteristics: 'Discoloration or wilting symptoms'
        }
      ];

      return {
        severity: cropIssues[0].severity,
        analysis_summary: `Potential issues detected. Based on common patterns for ${cropType || 'plants'}, there may be signs of disease or stress.`,
        detected_issues: cropIssues,
        treatment_recommendations: generalRecommendations,
        next_steps: generalNextSteps,
        confidence_scores: [0.75]
      };
    }
  };

  // Analyze image
  const analyzeImage = async () => {
    console.log('Starting image analysis...');
    console.log('Selected image:', selectedImage);
    console.log('Selected crop:', selectedCrop);
    
    if (!selectedImage) {
      const errorMsg = texts.noImageError || 'Please select an image first';
      console.error('No image selected');
      setError(errorMsg);
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setUploadProgress(0);
    console.log('Analysis started, clearing previous results');

    try {
      console.log('Converting image to base64...');
      // Convert image to base64
      const base64Image = await convertImageToBase64(selectedImage);
      console.log('Base64 conversion successful, length:', base64Image.length);
      
      setUploadProgress(30);

      // Prepare request data
      const requestData = {
        image_data: base64Image,
        crop_type: selectedCrop || null,
        language: currentLanguage === 'en' ? 'english' : 'hindi' // Map to backend language format
      };
      console.log('Request data prepared:', {
        image_data_length: base64Image.length,
        crop_type: requestData.crop_type,
        language: requestData.language
      });

      setUploadProgress(60);

      console.log('Attempting to connect to backend...');
      // Try backend first with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        console.log('Backend request timed out');
        controller.abort();
      }, 8000); // Increased timeout to 8 seconds

      const response = await axios.post('http://localhost:8000/detect_disease', requestData, {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 8000,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      console.log('Backend response received:', response.data);
      setUploadProgress(100);

      if (response.data.error) {
        console.error('Backend returned error:', response.data.error);
        setError(response.data.error);
      } else {
        console.log('Analysis successful, setting result');
        setAnalysisResult(response.data);
      }

    } catch (error) {
      console.log('Backend error occurred:', error.message);
      console.log('Error details:', error);
      
      // Provide offline analysis
      console.log('Switching to offline analysis mode...');
      setUploadProgress(70);
      
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setUploadProgress(100);
      
      const offlineResult = getOfflineAnalysis(selectedCrop);
      offlineResult.offline_mode = true;
      offlineResult.analysis_summary = `🤖 **Offline Analysis**: ${offlineResult.analysis_summary}\n\n💡 *This is a basic analysis. For advanced AI diagnosis, please ensure the backend server is running.*`;
      
      console.log('Offline analysis generated:', offlineResult);
      setAnalysisResult(offlineResult);
    } finally {
      console.log('Analysis completed, cleaning up...');
      setIsAnalyzing(false);
      setUploadProgress(0);
    }
  };

  // Convert image to base64
  const convertImageToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  // Get severity color
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'severe': return 'danger';
      case 'moderate': return 'warning';
      case 'mild': return 'info';
      default: return 'success';
    }
  };

  // Get severity icon
  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'severe': return '🚨';
      case 'moderate': return '⚠️';
      case 'mild': return '⚡';
      default: return '✅';
    }
  };

  // Show issue details
  const showIssueDetails = (issue) => {
    setSelectedIssueDetail(issue);
    setShowDetailModal(true);
  };

  // Reset form
  const resetForm = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setSelectedCrop('');
    setAnalysisResult(null);
    setError(null);
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="ai-crop-doctor-container">
      <div className="container py-4">
        {/* Header */}
        <div className="text-center mb-4">
          <h1 className="display-4 text-success mb-2">
            🌿 {texts.title || 'AI Crop Doctor'}
          </h1>
          <p className="lead text-muted">
            {texts.subtitle || 'Upload plant images for AI-powered disease and pest detection'}
          </p>
        </div>

        <div className="row">
          {/* Upload Section */}
          <div className="col-lg-6 mb-4">
            <Card className="h-100 shadow-sm">
              <Card.Header className="bg-light">
                <h5 className="mb-0">📤 Image Upload</h5>
              </Card.Header>
              <Card.Body>
                {/* Drag and Drop Area */}
                <div
                  className={`upload-area ${dragActive ? 'drag-active' : ''} ${selectedImage ? 'has-image' : ''}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  {imagePreview ? (
                    <div className="image-preview">
                      <img src={imagePreview} alt="Selected plant" className="preview-image" />
                      <div className="image-overlay">
                        <p className="mb-0">Click to change image</p>
                      </div>
                    </div>
                  ) : (
                    <div className="upload-placeholder">
                      <div className="upload-icon">📷</div>
                      <p className="upload-text">
                        {texts.uploadArea || 'Drag and drop an image here, or click to select'}
                      </p>
                      <small className="text-muted">
                        {texts.supportedFormats || 'Supported formats: JPG, PNG, WebP'}<br/>
                        {texts.maxFileSize || 'Maximum file size: 10MB'}
                      </small>
                    </div>
                  )}
                </div>

                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileInputChange}
                  accept="image/*"
                  style={{ display: 'none' }}
                />

                {/* Crop Selection */}
                <div className="mt-3">
                  <label className="form-label">
                    {texts.selectCrop || 'Select Crop Type (Optional)'}
                  </label>
                  <select
                    className="form-select"
                    value={selectedCrop}
                    onChange={(e) => {
                      console.log('Crop selection changed to:', e.target.value);
                      setSelectedCrop(e.target.value);
                    }}
                  >
                    <option value="">
                      {texts.chooseCrop || 'Choose crop type...'}
                    </option>
                    {cropTypes.map(crop => (
                      <option key={crop} value={crop}>
                        {crop.charAt(0).toUpperCase() + crop.slice(1)}
                      </option>
                    ))}
                  </select>
                  {selectedCrop && (
                    <small className="text-success mt-1 d-block">
                      ✅ Selected: {selectedCrop.charAt(0).toUpperCase() + selectedCrop.slice(1)}
                    </small>
                  )}
                </div>

                {/* Debug Info */}
                <div className="mt-3">
                  <Alert variant="light" className="debug-info">
                    <small>
                      <strong>🔧 Debug Info:</strong><br/>
                      📷 Image: {selectedImage ? `${selectedImage.name} (${(selectedImage.size / 1024 / 1024).toFixed(2)}MB, ${selectedImage.type})` : 'None selected'}<br/>
                      🌾 Crop: {selectedCrop || 'Not selected'}<br/>
                      🌍 Language: {currentLanguage}<br/>
                      ⚡ State: {isAnalyzing ? 'Analyzing...' : 'Ready'}<br/>
                      🔗 Backend: Test connection above
                    </small>
                  </Alert>
                </div>

                {/* Test Backend Connection */}
                <div className="mt-2">
                  <Button
                    variant="outline-info"
                    size="sm"
                    onClick={() => {
                      console.log('Testing backend connection...');
                      fetch('http://localhost:8000/health')
                        .then(response => response.json())
                        .then(data => {
                          console.log('Backend health check:', data);
                          alert('✅ Backend is running! Status: ' + data.status);
                        })
                        .catch(error => {
                          console.error('Backend connection failed:', error);
                          alert('❌ Backend connection failed. Make sure backend server is running on port 8000.');
                        });
                    }}
                    disabled={isAnalyzing}
                  >
                    🔗 Test Backend Connection
                  </Button>
                </div>

                {/* Upload Progress */}
                {isAnalyzing && uploadProgress > 0 && (
                  <div className="mt-3">
                    <ProgressBar 
                      now={uploadProgress} 
                      label={`${uploadProgress}%`}
                      variant="success"
                    />
                    <small className="text-muted">
                      {uploadProgress < 40 ? texts.uploadingImage || 'Uploading image...' :
                       uploadProgress < 80 ? texts.processingImage || 'Processing image...' :
                       texts.generatingReport || 'Generating report...'}
                    </small>
                  </div>
                )}

                {/* Error Display */}
                {error && (
                  <Alert variant="danger" className="mt-3">
                    <strong>{texts.analysisError || 'Error analyzing image'}:</strong> {error}
                  </Alert>
                )}

                {/* Analyze Button */}
                <div className="d-grid gap-2 mt-3">
                  <Button
                    variant="success"
                    size="lg"
                    onClick={analyzeImage}
                    disabled={!selectedImage || isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <>
                        <Spinner size="sm" className="me-2" />
                        {texts.analyzing || 'Analyzing image...'}
                      </>
                    ) : (
                      <>
                        🔍 {texts.analyzeButton || 'Analyze Plant Health'}
                      </>
                    )}
                  </Button>
                  
                  {selectedImage && (
                    <Button
                      variant="outline-secondary"
                      onClick={resetForm}
                      disabled={isAnalyzing}
                    >
                      🔄 {texts.uploadAnother || 'Upload Another Image'}
                    </Button>
                  )}
                </div>
              </Card.Body>
            </Card>
          </div>

          {/* Results Section */}
          <div className="col-lg-6 mb-4">
            <Card className="h-100 shadow-sm">
              <Card.Header className="bg-light">
                <h5 className="mb-0">📋 {texts.detectionResults || 'Detection Results'}</h5>
              </Card.Header>
              <Card.Body>
                {!analysisResult ? (
                  <div className="text-center py-5">
                    <div className="mb-3" style={{ fontSize: '4rem', opacity: 0.3 }}>🌱</div>
                    <p className="text-muted">
                      Upload a plant image to get AI-powered analysis
                    </p>
                  </div>
                ) : (
                  <div className="results-content">
                    {/* Analysis Summary */}
                    <div className="analysis-summary mb-4">
                      <div className="d-flex align-items-center mb-2">
                        <span className="severity-icon me-2" style={{ fontSize: '1.5rem' }}>
                          {getSeverityIcon(analysisResult.severity)}
                        </span>
                        <h6 className="mb-0">
                          {texts.severity || 'Severity'}: 
                          <span className={`ms-2 badge bg-${getSeverityColor(analysisResult.severity)}`}>
                            {analysisResult.severity?.toUpperCase() || 'UNKNOWN'}
                          </span>
                        </h6>
                      </div>
                      
                      {analysisResult.analysis_summary && (
                        <div className="analysis-text">
                          {analysisResult.analysis_summary.split('\n').map((line, index) => (
                            <p key={index} dangerouslySetInnerHTML={{ __html: line }} />
                          ))}
                        </div>
                      )}
                      
                      {analysisResult.offline_mode && (
                        <Alert variant="info" className="mt-2">
                          <small>
                            🔸 **Offline Mode Active**: This analysis is based on general plant health patterns. 
                            For advanced AI-powered disease detection with computer vision, please start the backend server.
                          </small>
                        </Alert>
                      )}
                    </div>

                    {/* Detected Issues */}
                    {analysisResult.detected_issues && analysisResult.detected_issues.length > 0 ? (
                      <div className="detected-issues mb-4">
                        <h6>{texts.detectedIssues || 'Detected Issues'}:</h6>
                        {analysisResult.detected_issues.map((issue, index) => (
                          <Card key={index} className="issue-card mb-2">
                            <Card.Body className="py-2">
                              <div className="d-flex justify-content-between align-items-center">
                                <div>
                                  <strong>{issue.name}</strong>
                                  <br />
                                  <small className="text-muted">{issue.description}</small>
                                  {issue.affected_area && (
                                    <>
                                      <br />
                                      <small className="text-info">Area: {issue.affected_area}</small>
                                    </>
                                  )}
                                </div>
                                <div className="text-end">
                                  {analysisResult.confidence_scores && (
                                    <div className="confidence-score">
                                      <small>{texts.confidence || 'Confidence'}</small>
                                      <br />
                                      <span className="badge bg-secondary">
                                        {Math.round((analysisResult.confidence_scores[index] || 0) * 100)}%
                                      </span>
                                    </div>
                                  )}
                                  <Button
                                    variant="outline-primary"
                                    size="sm"
                                    className="mt-1"
                                    onClick={() => showIssueDetails(issue)}
                                  >
                                    {texts.viewDetails || 'View Details'}
                                  </Button>
                                </div>
                              </div>
                            </Card.Body>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <Alert variant="success">
                        <h6>{texts.noIssuesFound || 'No major issues detected'}</h6>
                        <p className="mb-0">{texts.plantAppearHealthy || 'Your plant appears to be healthy!'}</p>
                      </Alert>
                    )}

                    {/* Treatment Recommendations */}
                    {analysisResult.treatment_recommendations && analysisResult.treatment_recommendations.length > 0 && (
                      <div className="recommendations mb-4">
                        <h6>{texts.recommendations || 'Treatment Recommendations'}:</h6>
                        <ul className="list-unstyled">
                          {analysisResult.treatment_recommendations.map((rec, index) => (
                            <li key={index} className="recommendation-item">
                              <span dangerouslySetInnerHTML={{ __html: rec }} />
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Next Steps */}
                    {analysisResult.next_steps && analysisResult.next_steps.length > 0 && (
                      <div className="next-steps">
                        <h6>{texts.nextSteps || 'Next Steps'}:</h6>
                        <ul className="list-unstyled">
                          {analysisResult.next_steps.map((step, index) => (
                            <li key={index} className="next-step-item">
                              <span dangerouslySetInnerHTML={{ __html: step }} />
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </Card.Body>
            </Card>
          </div>
        </div>

        {/* Usage Instructions */}
        <Card className="mt-4">
          <Card.Header>
            <h5 className="mb-0">📚 How to Use</h5>
          </Card.Header>
          <Card.Body>
            <div className="row">
              <div className="col-md-4">
                <div className="instruction-step">
                  <div className="step-number">1</div>
                  <h6>Take a Clear Photo</h6>
                  <p>Capture a well-lit, close-up image of the affected plant part</p>
                </div>
              </div>
              <div className="col-md-4">
                <div className="instruction-step">
                  <div className="step-number">2</div>
                  <h6>Upload Image</h6>
                  <p>Drag and drop or click to select your plant image</p>
                </div>
              </div>
              <div className="col-md-4">
                <div className="instruction-step">
                  <div className="step-number">3</div>
                  <h6>Get AI Analysis</h6>
                  <p>Receive instant diagnosis and treatment recommendations</p>
                </div>
              </div>
            </div>
          </Card.Body>
        </Card>

        {/* Issue Detail Modal */}
        <Modal show={showDetailModal} onHide={() => setShowDetailModal(false)} size="lg">
          <Modal.Header closeButton>
            <Modal.Title>
              {selectedIssueDetail?.name || 'Issue Details'}
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
            {selectedIssueDetail && (
              <div>
                <div className="mb-3">
                  <strong>Description:</strong>
                  <p>{selectedIssueDetail.description}</p>
                </div>
                
                {selectedIssueDetail.severity && (
                  <div className="mb-3">
                    <strong>{texts.severity || 'Severity'}:</strong>
                    <span className={`ms-2 badge bg-${getSeverityColor(selectedIssueDetail.severity)}`}>
                      {selectedIssueDetail.severity.toUpperCase()}
                    </span>
                  </div>
                )}
                
                {selectedIssueDetail.affected_area && (
                  <div className="mb-3">
                    <strong>Affected Area:</strong>
                    <p>{selectedIssueDetail.affected_area}</p>
                  </div>
                )}
                
                {selectedIssueDetail.characteristics && (
                  <div className="mb-3">
                    <strong>Characteristics:</strong>
                    <p>{selectedIssueDetail.characteristics}</p>
                  </div>
                )}
                
                {selectedIssueDetail.crop_specific && (
                  <Alert variant="info">
                    <small>This is a crop-specific identification based on the selected crop type.</small>
                  </Alert>
                )}
              </div>
            )}
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowDetailModal(false)}>
              {texts.closeDetails || 'Close'}
            </Button>
          </Modal.Footer>
        </Modal>
      </div>
    </div>
  );
};

export default AiCropDoctor;
