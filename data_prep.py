"""
Paddy Crop Disease Detection - Inference Script
Usage: python src/predict.py --image path/to/leaf.jpg
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

CLASSES = [
    "healthy",
    "leaf_blast",
    "brown_spot",
    "sheath_blight",
    "bacterial_blight"
]
IMG_SIZE = (128, 128)
MODEL_PATH = "models/paddy_cnn_model.h5"


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img


def predict(img_path, model_path=MODEL_PATH):
    model = tf.keras.models.load_model(model_path)
    img_array, img = preprocess_image(img_path)

    probs = model.predict(img_array)[0]
    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx] * 100

    print(f"\nPrediction  : {pred_class.replace('_', ' ').title()}")
    print(f"Confidence  : {confidence:.1f}%")
    print("\nAll class probabilities:")
    for cls, prob in zip(CLASSES, probs):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<20} {prob*100:5.1f}%  {bar}")

    # Show image with prediction
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class.replace('_', ' ').title()}\nConfidence: {confidence:.1f}%",
              fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("prediction_output.png")
    print("\nOutput image saved to prediction_output.png")

    return pred_class, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paddy disease predictor")
    parser.add_argument("--image", required=True, help="Path to leaf image")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to trained model")
    args = parser.parse_args()

    predict(args.image, args.model)
