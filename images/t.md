

## Core Application Files

### services/exchange-rate-service/app/main.py
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
import asyncio

from app.core.config import settings
from app.core.database import init_db
from app.api.v1 import rates
from app.middleware.auth import AuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.utils.logger import setup_logger
from app.tasks.rate_updater import RateUpdater

# Setup logging
logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Exchange Rate Service...")
    await init_db()
    
    # Start rate updater background task
    rate_updater = RateUpdater()
    update_task = asyncio.create_task(rate_updater.start_periodic_updates())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Exchange Rate Service...")
    update_task.cancel()
    try:
        await update_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="Payment Gateway - Exchange Rate Service",
    description="Real-time currency exchange rate service for international payments",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(rates.router, prefix="/api/v1/rates", tags=["Exchange Rates"])

@app.get("/health")
async def health_check():
    """Health check endpoint for service monitoring"""
    return {
        "status": "healthy", 
        "service": "exchange-rate-service",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "Payment Gateway Exchange Rate Service", 
        "version": "1.0.0",
        "description": "Real-time currency exchange rate service"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.SERVICE_PORT,
        reload=settings.DEBUG
    )
```

### services/exchange-rate-service/app/core/config.py
```python
import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://admin:admin123@localhost:5432/payment_gateway")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Exchange Rate API Configuration
    EXCHANGE_RATE_API_KEY: str = os.getenv("EXCHANGE_RATE_API_KEY", "your-api-key")
    EXCHANGE_RATE_API_URL: str = "https://v6.exchangerate-api.com/v6"
    BACKUP_API_URL: str = "https://api.fxratesapi.com/latest"
    
    # Rate Update Configuration
    RATE_UPDATE_INTERVAL: int = int(os.getenv("RATE_UPDATE_INTERVAL", "900"))  # 15 minutes
    RATE_CACHE_DURATION: int = int(os.getenv("RATE_CACHE_DURATION", "600"))   # 10 minutes
    
    # Supported Currencies
    SUPPORTED_CURRENCIES: List[str] = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "SGD"]
    BASE_CURRENCY: str = "BDT"
    
    # Service Configuration
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "exchange-rate-service")
    SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 1000
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Service Fees
    DEFAULT_SERVICE_FEE_PERCENTAGE: float = 2.0
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### services/exchange-rate-service/app/core/database.py
```python
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Database engine
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=StaticPool,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.DEBUG
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

async def init_db():
    """Initialize database tables"""
    try:
        # Import all models to ensure they are registered
        from app.models import exchange_rate
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def get_db() -> Session:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Session:
    """Get database session for service use"""
    return SessionLocal()
```

## Database Models

### services/exchange-rate-service/app/models/exchange_rate.py
```python
from sqlalchemy import Column, Integer, String, Numeric, DateTime, Boolean, Text, UniqueConstraint
from sqlalchemy.sql import func
from datetime import datetime

from app.core.database import Base

class ExchangeRate(Base):
    __tablename__ = "exchange_rates"
    
    id = Column(Integer, primary_key=True, index=True)
    currency_code = Column(String(3), nullable=False, index=True)
    rate_to_bdt = Column(Numeric(10, 4), nullable=False)
    source = Column(String(50), nullable=True)  # API source name
    last_updated = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Add unique constraint for currency_code and last_updated
    __table_args__ = (
        UniqueConstraint('currency_code', 'last_updated', name='_currency_time_uc'),
    )
    
    def __repr__(self):
        return f""
    
    def to_dict(self):
        """Convert exchange rate object to dictionary"""
        return {
            "id": self.id,
            "currency_code": self.currency_code,
            "rate_to_bdt": float(self.rate_to_bdt),
            "source": self.source,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def is_expired(self):
        """Check if the exchange rate has expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at.replace(tzinfo=None)

class RateUpdateLog(Base):
    __tablename__ = "rate_update_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    update_source = Column(String(50), nullable=False)
    currencies_updated = Column(Text, nullable=True)  # JSON string of updated currencies
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    error_details = Column(Text, nullable=True)
    update_duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f""
```

## Pydantic Schemas

### services/exchange-rate-service/app/schemas/rate.py
```python
from pydantic import BaseModel, validator
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

class RateRequest(BaseModel):
    currency_code: str
    
    @validator("currency_code")
    def validate_currency_code(cls, v):
        if len(v) != 3:
            raise ValueError("Currency code must be 3 characters")
        return v.upper()

class RateCalculationRequest(BaseModel):
    from_currency: str
    to_currency: str = "BDT"
    amount: Decimal
    service_fee_percentage: Optional[Decimal] = None
    
    @validator("from_currency", "to_currency")
    def validate_currency_codes(cls, v):
        if len(v) != 3:
            raise ValueError("Currency code must be 3 characters")
        return v.upper()
    
    @validator("amount")
    def validate_amount(cls, v):
        if v  100):
            raise ValueError("Service fee percentage must be between 0 and 100")
        return v

class ExchangeRateResponse(BaseModel):
    currency_code: str
    rate_to_bdt: Decimal
    source: Optional[str]
    last_updated: datetime
    expires_at: Optional[datetime]
    is_active: bool
    
    class Config:
        from_attributes = True

class RateCalculationResponse(BaseModel):
    original_amount: Decimal
    from_currency: str
    to_currency: str
    exchange_rate: Decimal
    converted_amount: Decimal
    service_fee_percentage: Decimal
    service_fee_amount: Decimal
    total_amount: Decimal
    calculation_time: datetime

class MultipleRatesResponse(BaseModel):
    rates: List[ExchangeRateResponse]
    base_currency: str = "BDT"
    last_updated: datetime

class RateHistoryResponse(BaseModel):
    currency_code: str
    rates: List[ExchangeRateResponse]
    period_start: datetime
    period_end: datetime

class RateUpdateStatus(BaseModel):
    currency_code: str
    status: str  # success, failed, skipped
    rate: Optional[Decimal] = None
    error: Optional[str] = None
    updated_at: datetime

class BulkRateUpdateResponse(BaseModel):
    update_id: str
    total_currencies: int
    successful_updates: int
    failed_updates: int
    update_status: List[RateUpdateStatus]
    update_duration_ms: int
    timestamp: datetime
```

## API Endpoints

### services/exchange-rate-service/app/api/v1/rates.py
```python
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime, timedelta

from app.core.database import get_db
from app.services.rate_service import RateService
from app.services.rate_fetcher import RateFetcher
from app.schemas.rate import (
    RateRequest, RateCalculationRequest, ExchangeRateResponse,
    RateCalculationResponse, MultipleRatesResponse, RateHistoryResponse,
    BulkRateUpdateResponse
)
from app.utils.response import SuccessResponse, ErrorResponse
from app.utils.exceptions import RateNotFoundError, ValidationError

router = APIRouter()

@router.get("/current", response_model=SuccessResponse[ExchangeRateResponse])
async def get_current_rate(
    currency: str = Query(..., description="Currency code (e.g., USD, EUR)"),
    db: Session = Depends(get_db)
):
    """Get current exchange rate for a specific currency"""
    try:
        rate_service = RateService(db)
        rate = await rate_service.get_current_rate(currency.upper())
        
        if not rate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Exchange rate not found for currency: {currency}"
            )
        
        return SuccessResponse(
            message="Exchange rate retrieved successfully",
            data=ExchangeRateResponse.from_orm(rate)
        )
    except RateNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve exchange rate"
        )

@router.get("/all", response_model=SuccessResponse[MultipleRatesResponse])
async def get_all_rates(
    db: Session = Depends(get_db)
):
    """Get current exchange rates for all supported currencies"""
    try:
        rate_service = RateService(db)
        rates = await rate_service.get_all_current_rates()
        
        return SuccessResponse(
            message="All exchange rates retrieved successfully",
            data=MultipleRatesResponse(
                rates=[ExchangeRateResponse.from_orm(rate) for rate in rates],
                last_updated=max(rate.last_updated for rate in rates) if rates else datetime.utcnow()
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve exchange rates"
        )

@router.post("/calculate", response_model=SuccessResponse[RateCalculationResponse])
async def calculate_amount(
    calculation_request: RateCalculationRequest,
    db: Session = Depends(get_db)
):
    """Calculate BDT amount with service fees for foreign currency amount"""
    try:
        rate_service = RateService(db)
        calculation = await rate_service.calculate_bdt_amount(calculation_request)
        
        return SuccessResponse(
            message="Amount calculated successfully",
            data=calculation
        )
    except RateNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate amount"
        )

@router.get("/history/{currency_code}", response_model=SuccessResponse[RateHistoryResponse])
async def get_rate_history(
    currency_code: str,
    days: int = Query(7, ge=1, le=365, description="Number of days of history"),
    db: Session = Depends(get_db)
):
    """Get exchange rate history for a specific currency"""
    try:
        rate_service = RateService(db)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        rates = await rate_service.get_rate_history(currency_code.upper(), start_date, end_date)
        
        return SuccessResponse(
            message="Rate history retrieved successfully",
            data=RateHistoryResponse(
                currency_code=currency_code.upper(),
                rates=[ExchangeRateResponse.from_orm(rate) for rate in rates],
                period_start=start_date,
                period_end=end_date
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve rate history"
        )

@router.post("/update", response_model=SuccessResponse[BulkRateUpdateResponse])
async def update_rates(
    background_tasks: BackgroundTasks,
    currencies: Optional[List[str]] = Query(None, description="Specific currencies to update"),
    force: bool = Query(False, description="Force update even if rates are fresh"),
    db: Session = Depends(get_db)
):
    """Manually trigger exchange rate updates"""
    try:
        rate_fetcher = RateFetcher(db)
        
        # Add update task to background
        background_tasks.add_task(
            rate_fetcher.update_rates_background,
            currencies or [],
            force
        )
        
        return SuccessResponse(
            message="Rate update initiated successfully",
            data={
                "update_id": f"manual-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "status": "initiated",
                "currencies": currencies or "all"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate rate update"
        )

@router.get("/health", response_model=SuccessResponse)
async def rate_service_health(
    db: Session = Depends(get_db)
):
    """Health check for rate service with data freshness"""
    try:
        rate_service = RateService(db)
        health_info = await rate_service.get_service_health()
        
        return SuccessResponse(
            message="Rate service health check",
            data=health_info
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )

@router.get("/compare", response_model=SuccessResponse)
async def compare_rates(
    base_currency: str = Query("USD", description="Base currency for comparison"),
    target_currencies: List[str] = Query(["EUR", "GBP", "CAD"], description="Currencies to compare"),
    amount: float = Query(1000.0, description="Amount to compare"),
    db: Session = Depends(get_db)
):
    """Compare exchange rates across multiple currencies"""
    try:
        rate_service = RateService(db)
        comparison = await rate_service.compare_currencies(base_currency, target_currencies, amount)
        
        return SuccessResponse(
            message="Currency comparison completed",
            data=comparison
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare currencies"
        )
```

## Service Layer

### services/exchange-rate-service/app/services/rate_service.py
```python
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from app.models.exchange_rate import ExchangeRate
from app.schemas.rate import RateCalculationRequest, RateCalculationResponse
from app.core.config import settings
from app.utils.exceptions import RateNotFoundError, ValidationError

logger = logging.getLogger(__name__)

class RateService:
    def __init__(self, db: Session):
        self.db = db
    
    async def get_current_rate(self, currency_code: str) -> Optional[ExchangeRate]:
        """Get the most recent active exchange rate for a currency"""
        rate = self.db.query(ExchangeRate).filter(
            and_(
                ExchangeRate.currency_code == currency_code,
                ExchangeRate.is_active == True
            )
        ).order_by(desc(ExchangeRate.last_updated)).first()
        
        if rate and rate.is_expired():
            logger.warning(f"Rate for {currency_code} has expired")
            # Still return the rate but log the expiration
        
        return rate
    
    async def get_all_current_rates(self) -> List[ExchangeRate]:
        """Get current exchange rates for all supported currencies"""
        rates = []
        for currency in settings.SUPPORTED_CURRENCIES:
            rate = await self.get_current_rate(currency)
            if rate:
                rates.append(rate)
        
        return rates
    
    async def calculate_bdt_amount(self, request: RateCalculationRequest) -> RateCalculationResponse:
        """Calculate BDT amount from foreign currency with service fees"""
        if request.from_currency == "BDT":
            raise ValidationError("Cannot convert from BDT to BDT")
        
        # Get current exchange rate
        rate = await self.get_current_rate(request.from_currency)
        if not rate:
            raise RateNotFoundError(f"Exchange rate not available for {request.from_currency}")
        
        # Calculate converted amount
        exchange_rate = rate.rate_to_bdt
        converted_amount = request.amount * exchange_rate
        
        # Calculate service fee
        service_fee_percentage = request.service_fee_percentage or Decimal(settings.DEFAULT_SERVICE_FEE_PERCENTAGE)
        service_fee_amount = converted_amount * (service_fee_percentage / 100)
        
        # Calculate total amount
        total_amount = converted_amount + service_fee_amount
        
        return RateCalculationResponse(
            original_amount=request.amount,
            from_currency=request.from_currency,
            to_currency=request.to_currency,
            exchange_rate=exchange_rate,
            converted_amount=converted_amount,
            service_fee_percentage=service_fee_percentage,
            service_fee_amount=service_fee_amount,
            total_amount=total_amount,
            calculation_time=datetime.utcnow()
        )
    
    async def get_rate_history(self, currency_code: str, start_date: datetime, end_date: datetime) -> List[ExchangeRate]:
        """Get exchange rate history for a currency within a date range"""
        rates = self.db.query(ExchangeRate).filter(
            and_(
                ExchangeRate.currency_code == currency_code,
                ExchangeRate.last_updated >= start_date,
                ExchangeRate.last_updated  Dict[str, Any]:
        """Get health information about the rate service"""
        health_info = {
            "status": "healthy",
            "currencies_supported": len(settings.SUPPORTED_CURRENCIES),
            "supported_currencies": settings.SUPPORTED_CURRENCIES,
            "rates_status": {}
        }
        
        for currency in settings.SUPPORTED_CURRENCIES:
            rate = await self.get_current_rate(currency)
            if rate:
                health_info["rates_status"][currency] = {
                    "available": True,
                    "last_updated": rate.last_updated.isoformat(),
                    "is_expired": rate.is_expired(),
                    "source": rate.source
                }
            else:
                health_info["rates_status"][currency] = {
                    "available": False,
                    "last_updated": None,
                    "is_expired": True,
                    "source": None
                }
        
        # Calculate overall health
        available_rates = sum(1 for status in health_info["rates_status"].values() if status["available"])
        if available_rates == 0:
            health_info["status"] = "critical"
        elif available_rates  Dict[str, Any]:
        """Compare exchange rates across multiple currencies"""
        if base_currency == "BDT":
            # Converting from BDT to other currencies
            comparisons = []
            for target_currency in target_currencies:
                rate = await self.get_current_rate(target_currency)
                if rate:
                    converted_amount = Decimal(amount) / rate.rate_to_bdt
                    comparisons.append({
                        "currency": target_currency,
                        "rate": float(rate.rate_to_bdt),
                        "converted_amount": float(converted_amount),
                        "available": True
                    })
                else:
                    comparisons.append({
                        "currency": target_currency,
                        "rate": None,
                        "converted_amount": None,
                        "available": False
                    })
        else:
            # Converting from foreign currency to BDT and other currencies
            base_rate = await self.get_current_rate(base_currency)
            if not base_rate:
                raise RateNotFoundError(f"Base currency rate not available: {base_currency}")
            
            bdt_amount = Decimal(amount) * base_rate.rate_to_bdt
            
            comparisons = [{
                "currency": "BDT",
                "rate": float(base_rate.rate_to_bdt),
                "converted_amount": float(bdt_amount),
                "available": True
            }]
            
            for target_currency in target_currencies:
                if target_currency == base_currency:
                    continue
                    
                target_rate = await self.get_current_rate(target_currency)
                if target_rate:
                    # Convert BDT to target currency
                    converted_amount = bdt_amount / target_rate.rate_to_bdt
                    comparisons.append({
                        "currency": target_currency,
                        "rate": float(target_rate.rate_to_bdt),
                        "converted_amount": float(converted_amount),
                        "available": True
                    })
                else:
                    comparisons.append({
                        "currency": target_currency,
                        "rate": None,
                        "converted_amount": None,
                        "available": False
                    })
        
        return {
            "base_currency": base_currency,
            "base_amount": amount,
            "comparisons": comparisons,
            "comparison_time": datetime.utcnow().isoformat()
        }
```

### services/exchange-rate-service/app/services/rate_fetcher.py
```python
import httpx
import logging
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json

from app.models.exchange_rate import ExchangeRate, RateUpdateLog
from app.core.config import settings
from app.utils.cache import CacheManager

logger = logging.getLogger(__name__)

class RateFetcher:
    def __init__(self, db: Session):
        self.db = db
        self.cache = CacheManager()
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def fetch_rates_from_api(self, source: str = "primary") -> Dict[str, float]:
        """Fetch exchange rates from external API"""
        try:
            if source == "primary":
                url = f"{settings.EXCHANGE_RATE_API_URL}/{settings.EXCHANGE_RATE_API_KEY}/latest/BDT"
            else:
                url = f"{settings.BACKUP_API_URL}?base=BDT"
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if source == "primary":
                # ExchangeRate-API format
                if data.get("result") != "success":
                    raise Exception(f"API returned error: {data.get('error-type')}")
                
                rates = {}
                conversion_rates = data.get("conversion_rates", {})
                
                for currency in settings.SUPPORTED_CURRENCIES:
                    if currency in conversion_rates:
                        # Convert to rate_to_bdt (inverse of the rate from BDT)
                        rates[currency] = 1 / conversion_rates[currency]
                
            else:
                # Backup API format
                rates = {}
                api_rates = data.get("rates", {})
                
                for currency in settings.SUPPORTED_CURRENCIES:
                    if currency in api_rates:
                        rates[currency] = 1 / api_rates[currency]
            
            return rates
            
        except Exception as e:
            logger.error(f"Failed to fetch rates from {source} API: {e}")
            raise
    
    async def update_single_rate(self, currency_code: str, rate_value: float, source: str) -> bool:
        """Update a single exchange rate in the database"""
        try:
            # Check if we need to update (rate is older than cache duration)
            existing_rate = self.db.query(ExchangeRate).filter(
                ExchangeRate.currency_code == currency_code,
                ExchangeRate.is_active == True
            ).order_by(ExchangeRate.last_updated.desc()).first()
            
            cache_duration = timedelta(seconds=settings.RATE_CACHE_DURATION)
            now = datetime.utcnow()
            
            if existing_rate and (now - existing_rate.last_updated.replace(tzinfo=None))  Dict[str, Any]:
        """Update all supported currency rates"""
        start_time = datetime.utcnow()
        update_log = {
            "successful_updates": 0,
            "failed_updates": 0,
            "currencies_updated": [],
            "errors": []
        }
        
        try:
            # Try primary API first
            try:
                rates = await self.fetch_rates_from_api("primary")
                source = "ExchangeRate-API"
            except Exception as e:
                logger.warning(f"Primary API failed: {e}, trying backup")
                try:
                    rates = await self.fetch_rates_from_api("backup")
                    source = "FxRatesAPI"
                except Exception as backup_error:
                    logger.error(f"Both APIs failed. Backup error: {backup_error}")
                    raise Exception("All rate sources unavailable")
            
            # Update each currency rate
            for currency_code, rate_value in rates.items():
                try:
                    if force or await self._should_update_rate(currency_code):
                        success = await self.update_single_rate(currency_code, rate_value, source)
                        if success:
                            update_log["successful_updates"] += 1
                            update_log["currencies_updated"].append(currency_code)
                        else:
                            update_log["failed_updates"] += 1
                            update_log["errors"].append(f"Failed to update {currency_code}")
                except Exception as e:
                    update_log["failed_updates"] += 1
                    update_log["errors"].append(f"Error updating {currency_code}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to update rates: {e}")
            update_log["errors"].append(f"General update error: {str(e)}")
        
        # Log the update
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        await self._log_update(source, update_log, duration_ms)
        
        update_log["update_duration_ms"] = duration_ms
        update_log["source"] = source
        
        return update_log
    
    async def update_rates_background(self, currencies: List[str], force: bool = False):
        """Background task for updating rates"""
        try:
            if currencies:
                # Update specific currencies
                for currency in currencies:
                    if currency in settings.SUPPORTED_CURRENCIES:
                        rates = await self.fetch_rates_from_api("primary")
                        if currency in rates:
                            await self.update_single_rate(currency, rates[currency], "Manual Update")
            else:
                # Update all rates
                await self.update_all_rates(force)
                
        except Exception as e:
            logger.error(f"Background rate update failed: {e}")
    
    async def _should_update_rate(self, currency_code: str) -> bool:
        """Check if a rate should be updated based on age"""
        existing_rate = self.db.query(ExchangeRate).filter(
            ExchangeRate.currency_code == currency_code,
            ExchangeRate.is_active == True
        ).order_by(ExchangeRate.last_updated.desc()).first()
        
        if not existing_rate:
            return True
        
        cache_duration = timedelta(seconds=settings.RATE_CACHE_DURATION)
        return (datetime.utcnow() - existing_rate.last_updated.replace(tzinfo=None)) >= cache_duration
    
    async def _log_update(self, source: str, update_log: Dict[str, Any], duration_ms: int):
        """Log the rate update operation"""
        try:
            log_entry = RateUpdateLog(
                update_source=source,
                currencies_updated=json.dumps(update_log["currencies_updated"]),
                success_count=update_log["successful_updates"],
                error_count=update_log["failed_updates"],
                error_details=json.dumps(update_log["errors"]) if update_log["errors"] else None,
                update_duration_ms=duration_ms
            )
            
            self.db.add(log_entry)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to log rate update: {e}")
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
```

## Background Tasks

### services/exchange-rate-service/app/tasks/rate_updater.py
```python
import asyncio
import logging
from datetime import datetime

from app.core.database import get_db_session
from app.services.rate_fetcher import RateFetcher
from app.core.config import settings

logger = logging.getLogger(__name__)

class RateUpdater:
    def __init__(self):
        self.running = False
        self.update_interval = settings.RATE_UPDATE_INTERVAL
    
    async def start_periodic_updates(self):
        """Start the periodic rate update task"""
        self.running = True
        logger.info(f"Starting periodic rate updates every {self.update_interval} seconds")
        
        while self.running:
            try:
                await self.update_rates()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                logger.info("Rate updater task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in periodic rate update: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(60)
    
    async def update_rates(self):
        """Update exchange rates"""
        try:
            db = get_db_session()
            rate_fetcher = RateFetcher(db)
            
            logger.info("Starting scheduled rate update")
            update_result = await rate_fetcher.update_all_rates()
            
            logger.info(
                f"Rate update completed: "
                f"{update_result['successful_updates']} successful, "
                f"{update_result['failed_updates']} failed"
            )
            
            await rate_fetcher.close()
            db.close()
            
        except Exception as e:
            logger.error(f"Failed to update rates: {e}")
    
    def stop(self):
        """Stop the periodic updates"""
        self.running = False
        logger.info("Stopping periodic rate updates")
```

## Utility Classes

### services/exchange-rate-service/app/utils/cache.py
```python
import redis
import json
import logging
from typing import Optional, Any
from datetime import datetime, timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.enabled = True
        except Exception as e:
            logger.warning(f"Redis connection failed, caching disabled: {e}")
            self.redis_client = None
            self.enabled = False
    
    async def get_rate(self, currency_code: str) -> Optional[dict]:
        """Get cached exchange rate"""
        if not self.enabled:
            return None
        
        try:
            key = f"rate:{currency_code}"
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                return json.loads(cached_data)
            
        except Exception as e:
            logger.warning(f"Failed to get cached rate for {currency_code}: {e}")
        
        return None
    
    async def set_rate(self, currency_code: str, rate_value: float, source: str):
        """Cache exchange rate"""
        if not self.enabled:
            return
        
        try:
            key = f"rate:{currency_code}"
            data = {
                "currency_code": currency_code,
                "rate_to_bdt": rate_value,
                "source": source,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            # Cache for rate cache duration
            self.redis_client.setex(
                key,
                settings.RATE_CACHE_DURATION,
                json.dumps(data)
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache rate for {currency_code}: {e}")
    
    async def invalidate_rate(self, currency_code: str):
        """Invalidate cached rate"""
        if not self.enabled:
            return
        
        try:
            key = f"rate:{currency_code}"
            self.redis_client.delete(key)
            
        except Exception as e:
            logger.warning(f"Failed to invalidate cache for {currency_code}: {e}")
    
    async def get_all_cached_rates(self) -> dict:
        """Get all cached rates"""
        if not self.enabled:
            return {}
        
        try:
            keys = self.redis_client.keys("rate:*")
            rates = {}
            
            for key in keys:
                currency_code = key.split(":")[1]
                cached_data = self.redis_client.get(key)
                if cached_data:
                    rates[currency_code] = json.loads(cached_data)
            
            return rates
            
        except Exception as e:
            logger.warning(f"Failed to get all cached rates: {e}")
            return {}
```

### services/exchange-rate-service/app/utils/exceptions.py
```python
class ExchangeRateError(Exception):
    """Base exception for exchange rate service"""
    pass

class RateNotFoundError(ExchangeRateError):
    """Raised when exchange rate is not found"""
    pass

class ValidationError(ExchangeRateError):
    """Raised when validation fails"""
    pass

class APIError(ExchangeRateError):
    """Raised when external API calls fail"""
    pass

class CacheError(ExchangeRateError):
    """Raised when cache operations fail"""
    pass
```

### services/exchange-rate-service/app/utils/response.py
```python
from pydantic import BaseModel
from typing import TypeVar, Generic, Optional, Any
from datetime import datetime

T = TypeVar('T')

class SuccessResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str
    data: Optional[T] = None
    timestamp: datetime = datetime.utcnow()

class ErrorResponse(BaseModel):
    success: bool = False
    error: dict
    timestamp: datetime = datetime.utcnow()
```

### services/exchange-rate-service/app/utils/logger.py
```python
import logging
import sys
from app.core.config import settings

def setup_logger(name: str) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    return logger
```

## Middleware

### services/exchange-rate-service/app/middleware/auth.py
```python
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for request logging and validation"""
    
    async def dispatch(self, request: Request, call_next):
        # Log incoming request
        logger.info(f"Incoming request: {request.method} {request.url}")
        
        # Add request ID for tracing
        import uuid
        request.state.request_id = str(uuid.uuid4())
        
        response = await call_next(request)
        
        # Log response
        logger.info(f"Response status: {response.status_code}")
        
        return response
```

### services/exchange-rate-service/app/middleware/rate_limit.py
```python
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict
from app.core.config import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time = settings.RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Add current request
        self.requests[client_ip].append(now)
        
        response = await call_next(request)
        return response
```

## Configuration Files

### services/exchange-rate-service/Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY shared/ ./shared/

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### services/exchange-rate-service/requirements.txt
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.7
alembic==1.12.1
pydantic[email]==2.5.0
pydantic-settings==2.0.3
httpx==0.25.2
redis==5.0.1
python-dotenv==1.0.0
asyncio-mqtt==0.11.1
celery[redis]==5.3.4
```

### services/exchange-rate-service/.env.example
```env
# Database
DATABASE_URL=postgresql://admin:admin123@postgres:5432/payment_gateway

# Redis
REDIS_URL=redis://redis:6379

# Exchange Rate API Configuration
EXCHANGE_RATE_API_KEY=your-exchange-rate-api-key
EXCHANGE_RATE_API_URL=https://v6.exchangerate-api.com/v6
BACKUP_API_URL=https://api.fxratesapi.com/latest

# Rate Update Configuration
RATE_UPDATE_INTERVAL=900
RATE_CACHE_DURATION=600

# Service Configuration
SERVICE_NAME=exchange-rate-service
SERVICE_PORT=8000
DEBUG=false

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60

# Service Fees
DEFAULT_SERVICE_FEE_PERCENTAGE=2.0
```

This comprehensive Exchange Rate Service implementation provides:

1. **Real-time Rate Fetching**: Fetches current exchange rates from multiple external APIs with fallback support
2. **Intelligent Caching**: Redis-based caching with configurable duration to reduce API calls
3. **Automatic Updates**: Background task that periodically updates exchange rates
4. **Rate Calculations**: Service fee calculations for international payments
5. **Historical Data**: Rate history tracking and retrieval
6. **Health Monitoring**: Service health checks with rate freshness indicators
7. **Currency Comparison**: Multi-currency comparison functionality
8. **Error Handling**: Robust error handling with fallback mechanisms
9. **Performance Optimization**: Database indexing and query optimization
10. **Comprehensive Logging**: Detailed logging for monitoring and debugging

The service integrates seamlessly with the payment gateway system, providing accurate exchange rates for international payment processing while maintaining high availability and performance.
