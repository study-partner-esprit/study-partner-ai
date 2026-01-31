"""Signal processor for handling incoming user signals."""
from typing import Dict, Any
from models.session import SessionRequest
from config.constants import (
    SIGNAL_START_SESSION,
    SIGNAL_PAUSE_SESSION,
    SIGNAL_COMPLETE_TASK,
    SIGNAL_REQUEST_HELP,
    SIGNAL_PROGRESS_UPDATE
)
import logging

logger = logging.getLogger(__name__)


class SignalProcessor:
    """Service for processing and validating incoming signals."""
    
    def __init__(self):
        """Initialize signal processor."""
        self.valid_signals = {
            SIGNAL_START_SESSION,
            SIGNAL_PAUSE_SESSION,
            SIGNAL_COMPLETE_TASK,
            SIGNAL_REQUEST_HELP,
            SIGNAL_PROGRESS_UPDATE
        }
        
    def validate_signal(self, signal_type: str) -> bool:
        """Validate if signal type is recognized.
        
        Args:
            signal_type: Type of signal to validate
            
        Returns:
            True if valid, False otherwise
        """
        is_valid = signal_type in self.valid_signals
        if not is_valid:
            logger.warning(f"Unknown signal type: {signal_type}")
        return is_valid
        
    def process_signal(
        self,
        signal_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process and enrich signal data.
        
        Args:
            signal_type: Type of signal
            context: Signal context data
            
        Returns:
            Processed signal data
        """
        if not self.validate_signal(signal_type):
            raise ValueError(f"Invalid signal type: {signal_type}")
            
        # Enrich signal with additional processing
        processed_data = {
            "signal_type": signal_type,
            "context": context,
            "requires_planning": self._requires_planning(signal_type),
            "requires_coaching": self._requires_coaching(signal_type),
            "requires_evaluation": self._requires_evaluation(signal_type)
        }
        
        logger.info(f"Processed signal: {signal_type}")
        return processed_data
        
    def create_session_request(
        self,
        user_id: str,
        signal_type: str,
        context: Dict[str, Any],
        session_id: str = None
    ) -> SessionRequest:
        """Create a SessionRequest from signal data.
        
        Args:
            user_id: User identifier
            signal_type: Type of signal
            context: Signal context
            session_id: Optional existing session ID
            
        Returns:
            SessionRequest object
        """
        return SessionRequest(
            user_id=user_id,
            signal_type=signal_type,
            session_id=session_id,
            context=context
        )
        
    def _requires_planning(self, signal_type: str) -> bool:
        """Check if signal requires planner agent.
        
        Args:
            signal_type: Type of signal
            
        Returns:
            True if planning is needed
        """
        return signal_type in {SIGNAL_START_SESSION, SIGNAL_PROGRESS_UPDATE}
        
    def _requires_coaching(self, signal_type: str) -> bool:
        """Check if signal requires coach agent.
        
        Args:
            signal_type: Type of signal
            
        Returns:
            True if coaching is needed
        """
        return signal_type in {SIGNAL_REQUEST_HELP, SIGNAL_PROGRESS_UPDATE}
        
    def _requires_evaluation(self, signal_type: str) -> bool:
        """Check if signal requires evaluator agent.
        
        Args:
            signal_type: Type of signal
            
        Returns:
            True if evaluation is needed
        """
        return signal_type in {SIGNAL_COMPLETE_TASK, SIGNAL_PROGRESS_UPDATE}
