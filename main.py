"""FastAPI application entry point."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from config.settings import get_settings
from orchestrator.meta_agent import MetaAgent
from models.session import SessionRequest, SessionResponse
from services.database import DatabaseService
from services.ai_logger import AILogger
from services.signal_processor import SignalProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize services
db_service = DatabaseService(settings.database_url)
ai_logger = AILogger()
signal_processor = SignalProcessor()
meta_agent = MetaAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Study Partner AI service...")
    await db_service.connect()
    logger.info("Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Study Partner AI service...")
    await db_service.disconnect()
    logger.info("Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.version
    }


@app.post(f"{settings.api_prefix}/session", response_model=SessionResponse)
async def process_session_request(request: SessionRequest):
    """Process a session request through the meta agent.
    
    Args:
        request: Session request containing signal and context
        
    Returns:
        Session response with agent decision
    """
    try:
        # Log the incoming signal
        ai_logger.log_signal(
            signal_type=request.signal_type,
            user_id=request.user_id,
            context=request.context,
            session_id=request.session_id
        )
        
        # Validate and process signal
        signal_processor.validate_signal(request.signal_type)
        
        # Process request through meta agent
        decision = await meta_agent.process_request(request)
        
        # Log the decision
        ai_logger.log_decision(
            agent_type=decision.agent_type,
            decision_type=decision.decision_type,
            decision_content=decision.content,
            confidence=decision.confidence,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        # Save decision to database
        decision_id = await db_service.save_decision(decision.dict())
        
        # Create response
        session_id = request.session_id or await db_service.save_session({
            "user_id": request.user_id,
            "signal_type": request.signal_type
        })
        
        response = SessionResponse(
            session_id=session_id,
            user_id=request.user_id,
            agent_type=decision.agent_type,
            decision_type=decision.decision_type,
            content=decision.content,
            confidence=decision.confidence
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        ai_logger.log_error(
            error_type="processing_error",
            error_message=str(e),
            context={"request": request.dict()},
            user_id=request.user_id,
            session_id=request.session_id
        )
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
