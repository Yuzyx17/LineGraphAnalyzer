import cv2
import numpy as np
from constants import *
from utils import rescale
from tensorflow.python.keras.models import load_model
# Load the trained model
model = load_model(MODEL)

# Load and preprocess the new image
image_index = 1
image_path = f"{IMAGE_PATH}{FILE_NAME}{image_index}{IMAGE_EXT}"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
image_height, image_width, _ = image.shape
test_image = cv2.resize(image, (224, 224))  # Resize to match VGG16 input shape
image = image / 255.0  # Normalize pixel values

# Make a prediction
prediction = model.predict(np.expand_dims(test_image, axis=0))
def rescale(value, original_range, target_range):
    original_min, original_max = original_range
    target_min, target_max = target_range
    
    # Handle the special case of 0
    if value == 0 and original_min == 0:
        return target_min
    
    # Perform the rescaling
    normalized_value = (value - original_min) / (original_max - original_min)
    rescaled_value = target_min + normalized_value * (target_max - target_min)
    
    return rescaled_value
# Denormalize the predicted coordinates
xmin = rescale(prediction[0][0]*224, [0, 224], [0, image_width])
ymin = rescale(prediction[0][1]*224, [0, 224], [0, image_height])
xmax = rescale(prediction[0][2]*224, [0, 224], [0, image_width])
ymax = rescale(prediction[0][3]*224, [0, 224], [0, image_height])
print("Converted ", xmin, ymin, xmax, ymax)
print("Predicted ", prediction[0][0], prediction[0][1], prediction[0][2], prediction[0][3])
# Display the predicted ROI on the image
image_with_roi = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)
cv2.imshow('Image with ROI', image_with_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
