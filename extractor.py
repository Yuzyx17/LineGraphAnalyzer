import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import rescale
from constants import *
"""
THIS FILE IS FOR EXTRACTING THE CANVAS
mode = 0 -> image processing
mode = 1 -> VGG16
"""

class Canvas:
    def __init__(self, image, mode=0):
        self.image = image
        self.canvas = None
        self.mode = mode
        self.x_range = None
        self.y_range = None
        self.color = cv2.cvtColor(self.image, cv2.COLOR_BGRA2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width, self.channels = self.color.shape
        self.bounding_rect = None
        self.bb_x, self.bb_y, self.bb_w, self.bb_h = (None, None, None, None)
        if not mode:
            self.set_bb_process()
        else:
            self.set_bb_vgg()
    
    def set_bb_process(self):
        _, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            self.bb_x, self.bb_y, self.bb_w, self.bb_h = cv2.boundingRect(contour)
            self.bb_x += 5
            self.bb_y += 0
            self.bb_w += -6 + self.bb_x 
            self.bb_h += -6 + self.bb_y
            self.bounding_rect = (self.bb_x, self.bb_y, self.bb_w, self.bb_h)
            self.canvas = self.image[self.bb_y:self.bb_h, self.bb_x:self.bb_w]
        else:
            raise("No Bounding Box Found using Image Pre-Processing")
    
    def set_bb_vgg(self):
        from tensorflow.python.keras.models import load_model
        # Load the trained model
        model = load_model(MODEL)

        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        image_height, image_width, _ = image.shape
        test_image = cv2.resize(image, (224, 224))  # Resize to match VGG16 input shape

        # Make a prediction
        prediction = model.predict(np.expand_dims(test_image, axis=0))

        # Denormalize the predicted coordinates
        self.bb_x = int(rescale(prediction[0][0]*224, [0, 224], [0, image_width]))
        self.bb_y = int(rescale(prediction[0][1]*224, [0, 224], [0, image_height]))
        self.bb_w = int(rescale(prediction[0][2]*224, [0, 224], [0, image_width]))
        self.bb_h = int(rescale(prediction[0][3]*224, [0, 224], [0, image_height]))

        print("Converted ", self.bb_x, self.bb_y, self.bb_w, self.bb_h)
        print("Predicted ", prediction[0][0], prediction[0][1], prediction[0][2], prediction[0][3])
        # Display the predicted ROI on the image
        self.bounding_rect = (self.bb_x, self.bb_y, self.bb_w, self.bb_h)
        self.canvas = self.image[self.bb_y:self.bb_h, self.bb_x:self.bb_w]

    def showCanvas(self):
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(nrows=1, ncols=2)

        # Access each subplot by its index
        ax1 = axes[0]
        ax2 = axes[1]

        ax1.axis("off")
        ax1.imshow(self.image)
        ax2.axis("off")
        ax2.imshow(self.canvas)
        plt.tight_layout()
        plt.show()
