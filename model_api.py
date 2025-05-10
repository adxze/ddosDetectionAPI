from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import uvicorn
import os
import traceback
import secrets
from datetime import datetime, timedelta
import jwt
from typing import Optional
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="DDOS Detection API", description="Secured API for detecting network intrusions")

# Security configuration
security = HTTPBearer()
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API key management
API_KEYS = os.environ.get("API_KEYS", "default-api-key-123").split(",")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler for rate limiting
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"Rate limit exceeded: {exc.detail}"}
    )


# Load AI model
try:
    model_tuple = joblib.load('model.pkl')
    if isinstance(model_tuple, tuple) and len(model_tuple) == 2:
        grid_search, label_encoders = model_tuple
        model = grid_search.best_estimator_
        logger.info(f"Model loaded successfully! Type: {type(model).__name__}")
    else:
        model = model_tuple
        label_encoders = {}
        logger.info("Model loaded but no label encoders found")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    label_encoders = {}


# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str


class APIKey(BaseModel):
    api_key: str


# Token creation function
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Authorization dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        api_key = payload.get("sub")
        if api_key is None or api_key not in API_KEYS:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")
        return api_key
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=403, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=403, detail="Invalid authentication credentials")


# Root endpoint
@app.get("/")
async def root():
    """Check if API is running"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "online",
        "model": model_status,
        "message": "DDOS Detection API is running. Use /token endpoint to authenticate."
    }


# Authentication endpoint
@app.post("/token", response_model=Token)
@limiter.limit("5/minute")
async def login(request: Request, api_key_data: APIKey):
    if api_key_data.api_key not in API_KEYS:
        logger.warning(f"Invalid API key attempt from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": api_key_data.api_key}, expires_delta=access_token_expires
    )

    logger.info(f"Token created for API key: {api_key_data.api_key[:8]}...")
    return {"access_token": access_token, "token_type": "bearer"}


# Main prediction endpoint with security
@app.post("/predict_csv")
@limiter.limit("10/minute")
async def predict_csv(
        request: Request,
        file: UploadFile = File(...),
        api_key: str = Depends(verify_token)
):
    """Make predictions based on uploaded CSV file"""
    client_host = request.client.host
    logger.info(f"Prediction request from {client_host} for file {file.filename}")

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
        logger.info(f"Loaded CSV with shape: {df.shape}")

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
            'intrusion_percentage': float(round(intrusion_count / total_count * 100, 2)),
            'high_confidence_intrusions': int(sum(probabilities > 0.9)),
            'medium_confidence_intrusions': int(sum((probabilities > 0.7) & (probabilities <= 0.9))),
            'low_confidence_intrusions': int(sum((probabilities > 0.5) & (probabilities <= 0.7)))
        }

        # If we have true labels, calculate accuracy
        if y_true is not None:
            accuracy = (predictions == y_true).mean() * 100
            summary['accuracy'] = float(round(accuracy, 2))

        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        logger.info(f"Prediction completed for {file.filename} from {client_host}")

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
        stack_trace = traceback.format_exc()
        logger.error(f"ERROR from {client_host}: {error_msg}")
        logger.error(f"STACK TRACE: {stack_trace}")

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
        stack_trace = traceback.format_exc()
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"STACK TRACE: {stack_trace}")
        raise Exception(error_msg)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None
    }


# API documentation endpoint
@app.get("/api/docs")
async def api_documentation():
    """Returns API usage documentation"""
    return {
        "endpoints": {
            "/": "API status check",
            "/health": "Health check endpoint",
            "/token": "Get authentication token (POST)",
            "/predict_csv": "Upload CSV for prediction (POST, requires authentication)",
            "/api/docs": "API documentation (this endpoint)"
        },
        "authentication": {
            "method": "Bearer token",
            "steps": [
                "1. Send POST request to /token with API key",
                "2. Receive JWT token",
                "3. Include token in Authorization header for subsequent requests"
            ],
            "example": {
                "get_token": {
                    "url": "/token",
                    "method": "POST",
                    "body": {"api_key": "your-api-key"}
                },
                "use_token": {
                    "url": "/predict_csv",
                    "method": "POST",
                    "headers": {"Authorization": "Bearer your-token-here"},
                    "body": "multipart/form-data with CSV file"
                }
            }
        },
        "rate_limits": {
            "/token": "5 requests per minute",
            "/predict_csv": "10 requests per minute"
        }
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("model_api:app", host="0.0.0.0", port=port)