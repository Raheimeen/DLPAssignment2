from flask import Flask, request,render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the saved model
model = load_model('Q1.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to the input shape of the model
    image = cv2.resize(image, (224, 224))
    # Convert the pixel values to the range [0, 1]
    image = image / 255.0
    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)
    return image

# Create a Flask application
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

# Define an endpoint for making predictions on new images
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    # Read the image file using OpenCV
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Preprocess the image
    image = preprocess_image(image)
    # Make a prediction using the model
    predictions = model.predict(image)
    # Get the predicted class label
    predicted_class = np.argmax(predictions[0])
    # Return the predicted class label as JSON
    return jsonify({'class': int(predicted_class)})

# Run the Flask application
if __name__ == '__main__':
    app.run(port=5000)