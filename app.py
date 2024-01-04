from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)


# Load the trained model
model = load_model('trained_model.h5')  # Update with the actual path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    img_file = request.files['file']

    # Save the image temporarily
    img_path = 'temp_image.jpg'
    img_file.save(img_path)

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
    img_array /= 255.0  # Normalize pixel values

    # Make predictions
    prediction = model.predict(img_array)

    # Display the result
    result = "It's a dog!" if prediction[0, 0] >= 0.5 else "It's a cat!"

    # Remove the temporarily saved image
    os.remove(img_path)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
