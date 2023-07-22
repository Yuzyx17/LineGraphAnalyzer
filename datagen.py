import os
import cv2
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from constants import *
from utils import rescale

"""
THIS FILE IS FOR CREATING THE DATASET, IMPORTANT!! LINE GRAPH MUST BE MARKED (have a dot per point)
"""

# Number of line graph images to generate
num_images = NUM_DATA_GEN

# Create a directory to store the dataset
os.makedirs(IMAGE_PATH, exist_ok=True)

# Generate line graph images
for i in range(num_images):
    # Generate random data for the line graph
    x_pts = np.arange(0, (np.random.randint(5, 10)*10)+1)
    # y_pts = abs(np.random.randn(len(x_pts)))*np.random.randint(5, 100)
    y_pts = np.random.randint(12, 126, size=len(x_pts))

    # Set the size of the canvas randomly each time
    canvas_width = np.random.randint(5, 12)
    canvas_height = np.random.randint(4, np.random.randint(5, 10))

    # Create a figure with the random size
    fig, ax = plt.subplots(figsize=(canvas_width, canvas_height))

    # Plot line graph
    ax.plot(x_pts, y_pts, linestyle='-', color='blue', linewidth=1)
    ax.plot(x_pts, y_pts, marker='o', linestyle='', color='black', markersize=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Line Graph {}'.format(i+1))

    # obtain max and min of x and y
    min_y = min(y_pts)
    max_y = max(y_pts)
    min_x = min(x_pts)
    max_x = max(x_pts)

    #Generate Even Labels
    pts_y = []
    steps = (max_y-min_y) / 10
    for z in range(10):
        if int(steps*z) == 0:
            pts_y.append(min_y)
        else:
            pts_y.append(round(steps*z)+min_y)
    pts_y.append(max_y)
    plt.yticks(pts_y, [str(ypt) for ypt in pts_y])
    # print(max_y, max_y-min_y)
    pts_x = []
    steps = (max_x-min_x) / 10
    for z in range(10):
        if int(steps*z) == 0:
            pts_x.append(min_x)
        else:
            pts_x.append(round(steps*z)+min_x)
    pts_x.append(max_x)
    plt.xticks(pts_x, [str(xpt) for xpt in pts_x])
    # print(pts_y, pts_x)
    
    # Save line graph image in the dataset folder
    filename = IMG.format(i+1)
    plt.savefig(filename)
    plt.clf()
    plt.close()

    # Load the image
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the bounding rectangle on the original image
        x, y, w, h = cv2.boundingRect(largest_contour)
        result = cv2.rectangle(image.copy(), (x+5, y), (x + w-1, y + h-6), (0, 255, 0), 1)
        xmin = x+5
        ymin = y
        xmax = x+w-1
        ymax = y+h-6

    # Create the root element
    annotation = ET.Element("annotation")

    # Create the folder element
    folder = ET.SubElement(annotation, "folder")
    folder.text = "images"

    # Create the filename element
    output_filename = os.path.splitext(filename)[0] + ANNOT_EXT
    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = os.path.splitext(filename)[0] + IMAGE_EXT

    # Create the path element
    path = ET.SubElement(annotation, "path")
    path.text = os.path.abspath(IMAGE_PATH)

    # Create the source element
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "LineGraph"

    # Create the size element
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image.shape[1])
    height = ET.SubElement(size, "height")
    height.text = str(image.shape[0])
    depth = ET.SubElement(size, "depth")
    depth.text = str(image.shape[2])

    # Create the segmented element
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    # Create the object element
    object_ = ET.SubElement(annotation, "object")
    name = ET.SubElement(object_, "name")
    name.text = "cvs"
    pose = ET.SubElement(object_, "pose")
    pose.text = "Unspecified"
    truncated = ET.SubElement(object_, "truncated")
    truncated.text = "0"
    difficult = ET.SubElement(object_, "difficult")
    difficult.text = "0"
    bndbox = ET.SubElement(object_, "bndbox")
    xmin_elem = ET.SubElement(bndbox, "xmin")
    xmin_elem.text = str(xmin)
    ymin_elem = ET.SubElement(bndbox, "ymin")
    ymin_elem.text = str(ymin)
    xmax_elem = ET.SubElement(bndbox, "xmax")
    xmax_elem.text = str(xmax)
    ymax_elem = ET.SubElement(bndbox, "ymax")
    ymax_elem.text = str(ymax)
    # Create the points element
    points = ET.SubElement(object_, "points")
    # Find the minimum and maximum values of x and y
    x_min_value = np.min(x_pts)
    x_max_value = np.max(x_pts)
    y_min_value = np.min(y_pts)
    y_max_value = np.max(y_pts)

    # Add the x and y coordinates as individual point elements
    for point_x, point_y in zip(x_pts, y_pts):
        point = ET.SubElement(points, "point")
        # point_x = rescale(point_x, (x_min_value, x_max_value), (0, 100)) # REMOVE
        point_x_elem = ET.SubElement(point, "x")
        point_x_elem.text = str(point_x)
        # point_y = rescale(point_y, (y_min_value, y_max_value), (0, 100)) # REMOVE
        point_y_elem = ET.SubElement(point, "y")
        point_y_elem.text = str(point_y)
    # Create the XML tree
    tree = ET.ElementTree(annotation)

    # Save the XML annotation to a file
    tree.write(ANT.format(i+1))

print("Data Generated")