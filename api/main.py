"""FastAPI Application Entry Point for PhishGuard.

Provides asynchronous REST API endpoints for URL phishing evaluation (/predict)
and health status diagnostics (/health) with CORS support and error handling.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.prediction import predictor

# Configure server-side logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phishguard_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load ML artifacts ONCE at startup."""
    logger.info("Initializing PhishGuard API server...")
    try:
        predictor.load_artifacts()
        logger.info("Successfully loaded XGBoost production model and feature schema into memory.")
    except Exception as e:
        logger.error(f"Critical error loading model artifacts at startup: {e}")
        raise e
    yield
    logger.info("Shutting down PhishGuard API server.")


app = FastAPI(
    title="PhishGuard — Intelligent Phishing Website Detection API",
    description=(
        "Asynchronous RESTful API service for real-time URL feature extraction, "
        "XGBoost-powered classification, confidence scoring, and risk breakdown."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS Middleware for local development & dashboard integration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="API Health Diagnostics",
    description="Check operational status and verify that production model artifacts are loaded in memory.",
)
async def health_check():
    """Verify server status and artifact loading."""
    return HealthResponse(
        status="healthy",
        model="xgboost",
        model_loaded=predictor.is_loaded,
        feature_schema_loaded=predictor.is_loaded,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate URL for Phishing Risk",
    description=(
        "Accepts a URL string, extracts 38 deterministic numerical features, "
        "runs XGBoost model inference, and returns prediction, confidence, risk level, and evidence."
    ),
)
async def predict_url(payload: PredictionRequest):
    """Evaluate target URL and return risk assessment."""
    try:
        result = predictor.predict(payload.url)
        return PredictionResponse(**result)
    except ValueError as val_err:
        logger.warning(f"Invalid input URL rejected: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error(f"Unexpected prediction failure: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred during URL evaluation.",
        )
