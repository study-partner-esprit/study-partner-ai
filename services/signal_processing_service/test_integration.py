"""Test script for fatigue detection integration with signal processing service.

This script tests the complete flow:
1. Initialize SignalProcessingService
2. Capture a video frame
3. Get a signal snapshot with fatigue detection
4. Verify the snapshot contains both focus and fatigue data
"""

import sys
from pathlib import Path

# Add root to path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from services.signal_processing_service.service import SignalProcessingService
import cv2


def test_fatigue_integration():
    """Test the integrated fatigue detection system."""
    print("=" * 60)
    print("Testing Fatigue Detection Integration")
    print("=" * 60)

    # Initialize service
    print("\n1. Initializing SignalProcessingService...")
    service = SignalProcessingService()

    # Check if adapters are loaded
    print(f"   - Focus adapter loaded: {service.focus_adapter.is_model_loaded()}")
    print(f"   - Fatigue adapter loaded: {service.fatigue_adapter.is_model_loaded()}")

    # Test with mock data (no video frame)
    print("\n2. Testing with mock data (no video frame)...")
    snapshot_mock = service.get_current_signal_snapshot(user_id="test_user_123")
    print("   Mock snapshot created:")
    print(
        f"   - Focus: {snapshot_mock.focus_state} (score: {snapshot_mock.focus_score:.2f}, conf: {snapshot_mock.focus_confidence:.2f})"
    )
    print(
        f"   - Fatigue: {snapshot_mock.fatigue_state} (score: {snapshot_mock.fatigue_score:.2f}, conf: {snapshot_mock.fatigue_confidence:.2f})"
    )

    # Test with webcam (if available)
    print("\n3. Testing with webcam frame...")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("   ⚠ Webcam not available, skipping live test")
        else:
            ret, frame = cap.read()
            cap.release()

            if ret:
                print("   ✓ Frame captured: {}".format(frame.shape))

                snapshot_live = service.get_current_signal_snapshot(
                    user_id="test_user_123", video_frame=frame
                )

                print("   Live snapshot created:")
                print(
                    f"   - Focus: {snapshot_live.focus_state} (score: {snapshot_live.focus_score:.2f}, conf: {snapshot_live.focus_confidence:.2f})"
                )
                print(
                    f"   - Fatigue: {snapshot_live.fatigue_state} (score: {snapshot_live.fatigue_score:.2f}, conf: {snapshot_live.fatigue_confidence:.2f})"
                )
            else:
                print("   ⚠ Failed to capture frame")
    except Exception as e:
        print(f"   ⚠ Error during webcam test: {e}")

    # Test snapshot retrieval
    print("\n4. Testing snapshot retrieval...")
    latest = service.get_latest_snapshot("test_user_123")
    if latest:
        print("   ✓ Retrieved latest snapshot from {}".format(latest.timestamp))
        print(f"   - Focus: {latest.focus_state}")
        print(f"   - Fatigue: {latest.fatigue_state}")
    else:
        print("   ⚠ No snapshot found (MongoDB may not be running)")

    print("\n" + "=" * 60)
    print("✓ Integration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_fatigue_integration()
