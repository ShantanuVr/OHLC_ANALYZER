"""
Configuration settings for the OHLC Analyzer backend.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Data paths
    DATA_BASE_PATH: Path = Path("/Volumes/Extreme SSD/ohlcv_data")
    NORMALIZED_DATA_PATH: Path = DATA_BASE_PATH / "normalized"
    
    # Default parameters
    DEFAULT_SYMBOL: str = "XAUUSD"
    DEFAULT_TIMEFRAME: str = "1h"
    
    # API settings
    API_PREFIX: str = "/api"
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"


settings = Settings()

