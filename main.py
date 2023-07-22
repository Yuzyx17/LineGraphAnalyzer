from interpreter import *
from label import LabelExtractor
from point import PointExtractor
from extractor import Canvas
from constants import *
import cv2
"""
THIS FILE IS FOR TESTING AND RUNNING
"""
index = 2
image = cv2.imread(IMG.format(index))
canvas = Canvas(image, mode=0) 
labels = LabelExtractor(canvas)
points = PointExtractor(canvas)
labels.extractLabel()
points.x_range = (labels.min_x, labels.max_x)
points.y_range = (labels.min_y, labels.max_y)
canvas.x_range = points.y_range
canvas.y_range = points.x_range
points.extractPoints()
# print(labels.max_y, labels.min_y, labels.max_x, labels.min_x)
canvas.showCanvas()
labels.showCanvas()

actual = LineGraph()
predicted = LineGraph()
interpreter = Interpreter(actual, predicted)
# Read XML file
tree = ET.parse(ANT.format(index)) 
root = tree.getroot()

# Find the 'points' element
points_element = root.find('.//object/points')

# Extract point coordinates
actual_points = []
for point_element in points_element.findall('point'):
    x = float(point_element.find('x').text)
    y = float(point_element.find('y').text)
    actual_points.append([x, y])

# Print the points list
actual.set_points(actual_points)
predicted.set_points(points.points)
# print(actual_points)
# print(points.points)
interpreter.obtain_point_metrics()
interpreter.show_metrics()
actual.interpret_data()