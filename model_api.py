from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import uvicorn
import os
import traceback
import hashlib
import secrets
from typing import Optional

# Create FastAPI app
app = FastAPI(title="DDOS Detection API", description="Simplified secure API for detecting network intrusions")

# Simple API key management - set this in environment variable
API_KEYS = os.environ.get("API_KEYS", "default-api-key-123").split(",")
# Generate a random secret for this session if not provided
SECRET_SALT = os.environ.get("SECRET_SALT", secrets.token_hex(16))

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI model
try:
    model_tuple = joblib.load('model.pkl')
    if isinstance(model_tuple, tuple) and len(model_tuple) == 2:
        grid_search, label_encoders = model_tuple
        model = grid_search.best_estimator_
        print(f"Model loaded successfully! Type: {type(model).__name__}")
    else:
        model = model_tuple
        label_encoders = {}
        print("Model loaded but no label encoders found")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    label_encoders = {}

# Simple authentication function
def verify_api_key(api_key: str) -> bool:
    """Verify if the API key is valid"""
    return api_key in API_KEYS

# Authentication dependency
async def get_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key required"
        )
    if not verify_api_key(x_api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return x_api_key

# Root endpoint
@app.get("/")
async def root():
    """Check if API is running"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "online",
        "model": model_status,
        "message": "DDOS Detection API is running. Use X-API-Key header for authentication."
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Main prediction endpoint with security
@app.post("/predict_csv")
async def predict_csv(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Header(None, alias="X-API-Key")
):
    """Make predictions based on uploaded CSV file"""
    # Verify API key
    if api_key is None or not verify_api_key(api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    
    client_host = request.client.host
    print(f"Prediction request from {client_host} for file {file.filename}")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Limit file size (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    try:
        # Read file with size limit
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Save the uploaded file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            buffer.write(contents)
        
        # Read the CSV
        df = pd.read_csv(temp_file)
        print(f"Loaded CSV with shape: {df.shape}")
        
        # Drop unnecessary columns if they exist
        df = df.drop(['id', 'attack_cat'], axis=1, errors='ignore')
        
        # Prepare features
        X = df.copy()
        if 'label' in X.columns:
            y_true = X['label'].copy()
            X = X.drop('label', axis=1)
        else:
            y_true = None
        
        # Preprocess the data
        X_processed = preprocess_data(X)
        
        # Make predictions
        predictions = model.predict(X_processed)
        
        try:
            probabilities = model.predict_proba(X_processed)[:, 1]
        except:
            probabilities = predictions.astype(float)
        
        # Create result summary
        normal_count = int(sum(predictions == 0))
        intrusion_count = int(sum(predictions == 1))
        total_count = len(predictions)
        
        summary = {
            'total_connections': total_count,
            'normal_connections': normal_count,
            'intrusion_connections': intrusion_count,
            'intrusion_percentage': float(round(intrusion_count / total_count * 100, 2))
        }
        
        # If we have true labels, calculate accuracy
        if y_true is not None:
            accuracy = (predictions == y_true).mean() * 100
            summary['accuracy'] = float(round(accuracy, 2))
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        print(f"Prediction completed for {file.filename} from {client_host}")
        
        return {
            'success': True,
            'result_counts': {
                'Normal': normal_count,
                'Intrusion': intrusion_count
            },
            'summary': summary,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"ERROR from {client_host}: {error_msg}")
        
        # Cleanup
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        
        raise HTTPException(status_code=500, detail=error_msg)

def preprocess_data(df):
    """
    Preprocess data to match the format expected by the model
    """
    try:
        # Handle categorical columns with label encoding
        for col in df.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                # Handle unknown categories
                df[col] = df[col].astype(str)
                for val in df[col].unique():
                    if val not in label_encoders[col].classes_:
                        label_encoders[col].classes_ = np.append(label_encoders[col].classes_, val)
                df[col] = label_encoders[col].transform(df[col])
            else:
                df[col] = 0
                
        # Apply log transformations as your model was trained with
        if 'dload' in df.columns:
            df['dload'] = np.log1p(df['dload'])
            
        if 'ct_dst_sport_ltm' in df.columns:
            df['ct_dst_sport_ltm'] = np.log1p(df['ct_dst_sport_ltm'])
            
        if 'dmean' in df.columns:
            df['dmean'] = np.log1p(df['dmean'])
        
        return df
    except Exception as e:
        error_msg = f"Error preprocessing data: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise Exception(error_msg)

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# API documentation endpoint
@app.get("/api/docs")
async def api_documentation():
    """Returns API usage documentation"""
    return {
        "endpoints": {
            "/": "API status check",
            "/health": "Health check endpoint",
            "/predict_csv": "Upload CSV for prediction (POST, requires API key)",
            "/api/docs": "API documentation (this endpoint)"
        },
        "authentication": {
            "method": "API Key in header",
            "header_name": "X-API-Key",
            "example": {
                "headers": {
                    "X-API-Key": "your-api-key-here"
                }
            }
        },
        "usage_example": {
            "curl": 'curl -X POST "http://api-url/predict_csv" -H "X-API-Key: your-api-key" -F "file=@your-file.csv"'
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("model_api:app", host="0.0.0.0", port=port)
