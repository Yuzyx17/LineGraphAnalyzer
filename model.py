import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from keras.applications import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import load_model
from sklearn.model_selection import train_test_split
from constants import *
"""
THIS FILE IS FOR TRAINING OR MODELLING THE VGG/CNN
"""
# Set the paths to your dataset and annotations
dataset_path = IMAGE_PATH
annotations_path = ANNOT_PATH

# Define hyperparameters
num_channels = 3
num_coords = 4  # Number of coordinates for ROI (xmin, ymin, xmax, ymax)
batch_size = 24
epochs = 8
learning_rate = 0.001

# Initialize empty lists for images and annotations
images = []
annotations = []
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
# Load and preprocess the data
for annotation_file in os.listdir(annotations_path):
    if annotation_file.endswith(".xml"):
        annotation_path = os.path.join(annotations_path, annotation_file)
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        image_filename = root.find(".//filename").text

        # Load and preprocess the image
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        image_height, image_width, _ = image.shape
        image = cv2.resize(image, (224, 224))  # Resize to match VGG16 input shape
        images.append(image)

        # Extract the bounding box coordinates from the annotation
        xmin = int(root.find(".//xmin").text)
        ymin = int(root.find(".//ymin").text)
        xmax = int(root.find(".//xmax").text)
        ymax = int(root.find(".//ymax").text)

        # Normalize the bounding box coordinates
        xmin = rescale(xmin, [0, image_width], [0, 224]) /224
        ymin = rescale(ymin, [0, image_height], [0, 224])/224
        xmax = rescale(xmax, [0, image_width], [0, 224])/224
        ymax = rescale(ymax, [0, image_height], [0, 224])/224
        # Store the normalized bounding box coordinates
        annotations.append([xmin, ymin, xmax, ymax])

# Convert the lists to NumPy arrays
images = np.array(images)
annotations = np.array(annotations)

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, annotations, test_size=0.2, random_state=42)

# Load the pre-trained model
if os.path.exists(MODEL):
    model = load_model(MODEL)
    print("Pre-trained model loaded successfully.")
else:
    print("Pre-trained model not found. Training from scratch.")

    # Load the pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the pre-trained model
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)

    # Add a custom output layer for regression
    predictions = Dense(num_coords, name='predictions')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=predictions)

    # Model Optimizer
    optimizer = adam_v2.Adam(learning_rate=learning_rate)
    # Compile the model
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))
# Evaluate the model
loss, accuracy = model.evaluate(x_val, y_val)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)
 
# Save the model
model.save(MODEL)