# Local Backend Setup for AI-Assisted Farming Application

## Overview
This backend has been modified to run fully offline on your local machine instead of using cloud-based machine learning services. The application now loads models locally from files instead of making API calls to IBM Cloud Watson Machine Learning.

## Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

## Quick Setup

We've provided a setup script to help you get started quickly:

```
python setup.py
```

This script will:
1. Check your Python version
2. Install all required dependencies
3. Check if model files and label encoders exist
4. Provide next steps

## Manual Installation

If you prefer to set up manually:

1. Clone the repository (if you haven't already)

2. Navigate to the backend directory:
   ```
   cd backend
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Model Setup

Before running the application, you need to export your trained models from IBM Cloud and place them in the correct location:

1. Export your crop recommendation model from IBM Cloud Watson Machine Learning in joblib format
2. Export your yield prediction model from IBM Cloud Watson Machine Learning in joblib format
3. Place these models in the following locations:
   - Crop recommendation model: `static/models/crop_recommendation_model.joblib`
   - Yield prediction model: `static/models/yield_prediction_model.joblib`

## Running the Application

You have two options for running the application:

### Option 1: Flask (Original Implementation)

```
python app.py
```

The server will run at `http://localhost:8000`

### Option 2: FastAPI (Recommended)

We've also provided a FastAPI implementation which offers better performance, automatic API documentation, and more modern features:

```
python fastapi_app.py
```

The server will run at `http://localhost:8000`

You can access the interactive API documentation at `http://localhost:8000/docs`

### Direct Model Usage

If you want to use the models directly without starting a server, you can use the example script:

```
python example_local_inference.py
```

This script demonstrates how to load the models and make predictions programmatically.

## API Endpoints

### Crop Recommendation
Endpoint: `/crop_prediction`

Example request body:
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "ph": 6.5,
  "rainfall": 200,
  "state": "gujarat",
  "city": "ahmedabad"
}
```

### Yield Prediction
Endpoint: `/yield_prediction`

Example request body:
```json
{
  "state": "gujarat",
  "city": "amreli",
  "season": "kharif",
  "crop": "rice",
  "area": 120.12
}
```

## Troubleshooting

If you encounter any issues with model loading, check the following:

1. Ensure the model files are in the correct location
2. Verify that the model format is compatible with joblib.load()
3. Check the console for any error messages

## Dependencies

The main dependencies for this application are:
- Flask: Web framework
- scikit-learn: Machine learning library
- joblib: Model serialization/deserialization
- numpy: Numerical computing
- pandas: Data manipulation

All dependencies are listed in the `requirements.txt` file.