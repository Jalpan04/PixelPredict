import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64

# ---------- Flask App ----------
app = Flask(__name__)

# ---------- Model Utilities ----------
class SimpleNN:
    def __init__(self, W1_path, b1_path, W2_path, b2_path):
        self.W1 = np.load(W1_path)
        self.b1 = np.load(b1_path)
        self.W2 = np.load(W2_path)
        self.b2 = np.load(b2_path)

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def forward(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = self.softmax(Z2)
        return A2

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs)

# ---------- Load the Model Once ----------
model = SimpleNN('W1.npy', 'b1.npy', 'W2.npy', 'b2.npy')

# ---------- Image Preprocessing ----------
def preprocess_image(image_data_base64):
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_data_base64)
        image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28

        # Normalize and reshape
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(784, 1)
        return image_array
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {str(e)}")

# ---------- Routes ----------
@app.route('/')
def home():
    try:
        return open('index.html').read()
    except FileNotFoundError:
        return "<h1>index.html not found</h1>", 404

@app.route('/predict', methods=['POST'])
def predict_digit():
    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image = preprocess_image(data['image'])
        probs = model.forward(image)
        predicted_digit = int(np.argmax(probs))
        confidences = probs.flatten().tolist()  # <-- return all 10!

        return jsonify({'prediction': predicted_digit, 'confidences': confidences})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- Main ----------
if __name__ == "__main__":
    app.run(debug=True)
