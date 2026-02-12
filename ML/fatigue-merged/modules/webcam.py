import cv2
import time
import logging

# Suppress OpenCV warnings
logging.getLogger("libGL").setLevel(logging.ERROR)

class Webcam:
    """
    Enhanced webcam handler combining robust camera management
    from the original system with MediaPipe compatibility.
    """

    def __init__(self, camera_id=0, width=640, height=480, fps=15):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        print(f"Initializing webcam with id {camera_id}")

        # Try different camera backends for better compatibility
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]  # Linux V4L2 backend first, then any
        self.cap = None

        for backend in backends:
            self.cap = cv2.VideoCapture(camera_id, backend)
            if self.cap.isOpened():
                print(f"Successfully opened camera with backend: {backend}")
                break

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Failed to open camera with any backend")

        # Set camera properties for reliability
        # Force MJPG format (more reliable on Linux, avoids freezes)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size

        # Verify camera is working
        if not self.cap.isOpened():
            raise RuntimeError("Failed to configure camera properties")

        print("Camera initialized successfully")

        # Camera health monitoring
        self.last_successful_frame = time.time()
        self.frame_count = 0
        self.recovery_attempts = 0

    def get_frame(self):
        """
        Get a single frame with enhanced error handling and recovery.
        Returns: (frame_bgr, frame_rgb) or (None, None) on failure
        """
        ret, frame_bgr = self.cap.read()
        current_time = time.time()

        if not ret or frame_bgr is None:
            print(f"Camera read failed at frame {self.frame_count}")
            return self._attempt_recovery()

        self.frame_count += 1
        self.last_successful_frame = current_time

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        return frame_bgr, frame_rgb

    def _attempt_recovery(self):
        """
        Attempt to recover from camera failure.
        Returns: (frame_bgr, frame_rgb) or (None, None) if recovery fails
        """
        print("Attempting camera recovery...")
        self.recovery_attempts += 1

        # Try to grab a frame with timeout
        self.cap.grab()  # Try to grab without decoding
        ret, frame_bgr = self.cap.retrieve()

        if ret and frame_bgr is not None:
            print("Camera recovered via grab/retrieve")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return frame_bgr, frame_rgb

        # More aggressive recovery
        print("Camera recovery: releasing and reopening...")
        self.cap.release()
        time.sleep(0.5)

        # Try to reopen camera
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        for backend in backends:
            self.cap = cv2.VideoCapture(self.camera_id, backend)
            if self.cap.isOpened():
                # Reconfigure camera
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                print(f"Camera recovered with backend: {backend}")

                # Flush buffer after reopening
                for _ in range(10):
                    self.cap.read()

                self.last_successful_frame = time.time()
                self.recovery_attempts = 0

                # Try to get a frame immediately
                ret, frame_bgr = self.cap.read()
                if ret and frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    return frame_bgr, frame_rgb

        print("Failed to recover camera")
        return None, None

    def is_healthy(self, timeout=5.0):
        """
        Check if camera is healthy based on recent frame acquisition.
        """
        return (time.time() - self.last_successful_frame) < timeout

    def calibrate(self, num_frames=100):
        """
        Calibrate camera by capturing and discarding initial frames.
        Returns average processing time per frame.
        """
        print("Calibrating camera...")

        start_time = time.time()
        valid_frames = 0

        for i in range(num_frames):
            ret, frame_bgr = self.cap.read()
            if ret and frame_bgr is not None:
                valid_frames += 1
                if i % 20 == 0:
                    print(f"Calibration progress: {i+1}/{num_frames}")

        total_time = time.time() - start_time
        avg_time = total_time / max(valid_frames, 1)

        print(f"✅ Calibration complete. Avg frame time: {avg_time:.4f}s")
        return avg_time

    def show_loop(self, window_name="Camera Feed", exit_key='q'):
        """
        Continuously show the camera feed until exit key is pressed.
        """
        print(f"Starting camera preview. Press '{exit_key}' to exit.")

        while True:
            frame_bgr, _ = self.get_frame()
            if frame_bgr is None:
                print("Lost camera feed, exiting preview...")
                break

            cv2.imshow(window_name, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(exit_key):
                break

        self.release()

    def release(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released")

    def get_stats(self):
        """
        Get camera statistics.
        Returns: dict with camera stats
        """
        return {
            "frame_count": self.frame_count,
            "recovery_attempts": self.recovery_attempts,
            "last_frame_time": self.last_successful_frame,
            "is_healthy": self.is_healthy()
        }