"""
FastAPI application for OHLC Analyzer - No-Code Backtesting Platform.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router

app = FastAPI(
    title="OHLC Analyzer API",
    description="No-Code Backtesting Platform for XAUUSD Trading with Smart Money Concepts",
    version="1.0.0",
    debug=settings.DEBUG,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix=settings.API_PREFIX)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ohlc-analyzer"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "OHLC Analyzer API",
        "docs": "/docs",
        "health": "/health",
    }

