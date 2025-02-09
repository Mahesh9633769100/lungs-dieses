from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("pneumonia.keras")

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB (Ensures 3 channels)
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(file)
            input_image = preprocess_image(image)
            prediction = model.predict(input_image)
            
            # Assuming your model outputs probabilities, get the class with the highest probability
            class_names = ["Normal", "Viral Pneumonia"]  # Adjusted to match the class names
            predicted_class = class_names[np.argmax(prediction)]  # Get predicted class

            # Display predicted class and its probability (if needed)
            confidence = np.max(prediction)  # Confidence score for prediction
            return render_template("index.html", result=f"Prediction: {predicted_class} with {confidence*100:.2f}% confidence")
    
    return render_template("index.html", result="")

if __name__ == "__main__":
    app.run(debug=True)
