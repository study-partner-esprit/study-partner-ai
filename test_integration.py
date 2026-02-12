"""Quick test script to verify the Coach + ML integration.

This script tests the full pipeline without starting the HTTP service.
"""

import sys
from datetime import datetime

# Test imports
try:
    from services.signal_processing_service.service import SignalProcessingService
    from services.ai_orchestrator.orchestrator import AIOrchestrator
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_signal_service():
    """Test signal processing service."""
    print("\n🔍 Testing Signal Processing Service...")
    
    service = SignalProcessingService()
    
    # Test if service is ready
    if service.is_ready():
        print("✅ Signal service is ready")
    else:
        print("⚠️  Signal service not fully ready (models may not be loaded)")
    
    # Test signal collection
    try:
        snapshot = service.get_current_signal_snapshot("test_user_123")
        print(f"✅ Signal snapshot created:")
        print(f"   - User: {snapshot.user_id}")
        print(f"   - Focus State: {snapshot.focus_state}")
        print(f"   - Focus Score: {snapshot.focus_score:.2f}")
        print(f"   - Confidence: {snapshot.focus_confidence:.2f}")
        return True
    except Exception as e:
        print(f"❌ Signal collection failed: {e}")
        return False

def test_orchestrator():
    """Test AI orchestrator."""
    print("\n🔍 Testing AI Orchestrator...")
    
    try:
        orchestrator = AIOrchestrator()
        print("✅ Orchestrator initialized")
        
        # Test coach execution
        result = orchestrator.run_coach(
            user_id="test_user_123",
            current_time=datetime.now(),
            ignored_count=0,
            do_not_disturb=False
        )
        
        print(f"✅ Coach executed successfully:")
        print(f"   - Action: {result.action_type}")
        print(f"   - Message: {result.message}")
        print(f"   - Reasoning: {result.reasoning[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ML → Coach Integration Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Signal Service
    results.append(("Signal Service", test_signal_service()))
    
    # Test 2: Orchestrator
    results.append(("Orchestrator", test_orchestrator()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("1. Start the FastAPI service: python api_service.py")
        print("2. Test the HTTP endpoint: curl http://localhost:8000/health")
        print("3. Start the Express backend")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
