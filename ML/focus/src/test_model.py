import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Force CPU usage to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def test_focus_model():
    """Test the focus detection model with a sample image"""

    # Load the model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models', 'focus_model.h5')
    print(f"Loading model from: {model_path}")

    try:
        model = load_model(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Print model summary
    print("\nModel Summary:")
    model.summary()

    # Test with a sample image
    sample_image_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Engaged', 'engaged', '0001.jpg')
    print(f"\nTesting with sample image: {sample_image_path}")

    if not os.path.exists(sample_image_path):
        print("✗ Sample image not found")
        return

    # Load and preprocess image
    img = cv2.imread(sample_image_path)
    if img is None:
        print("✗ Could not load image")
        return

    print(f"Original image shape: {img.shape}")

    # Resize to model input size (224x224 as seen in the test scripts)
    img_resized = cv2.resize(img, (224, 224))
    print(f"Resized image shape: {img_resized.shape}")

    # Preprocess: expand dims and normalize
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]

    print(f"Input shape for model: {img_array.shape}")

    # Make prediction
    print("\nMaking prediction...")
    predictions = model.predict(img_array, verbose=0)

    # Define label mapping (from the test scripts)
    label_map = {0: "Focused", 1: "Drifting", 2: "Lost"}

    # Get prediction results
    pred_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    pred_label = label_map[pred_idx]

    print(f"Predictions: {predictions[0]}")
    print(f"Predicted class: {pred_idx} ({pred_label})")
    print(".2f")

    # Show all probabilities
    print("\nAll probabilities:")
    for i, prob in enumerate(predictions[0]):
        print(".4f")

    print("\n✓ Model test completed successfully!")

if __name__ == "__main__":
    test_focus_model()