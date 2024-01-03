import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('trained_model.h5')  # Use the actual path to your trained model

# Get input image path from the user
img_path = input("Enter the path to the image: ")

# Load and preprocess the image
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
img_array /= 255.0  # Normalize pixel values

# Make predictions
prediction = model.predict(img_array)

# Display the result
if prediction[0, 0] >= 0.5:
    print("It's a dog!")
else:
    print("It's a cat!")
