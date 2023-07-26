import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import rescale, rescale_points
from extractor import Canvas
"""
THIS FILE IS FOR EXTRACTING THE POINTS IN THE CANVAS
"""
class PointExtractor():
    def __init__(self, image: Canvas, show=False):
        self.canvas = image
        self.show = show
        self.points = []
        self.y_range = (0, 100)
        self.x_range = (0, 100)
        self.dilated_canvas = self.dilatedLines()

    def dilatedLines(self):
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.canvas.canvas, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to create a binary image with only black pixels
        _, binary = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY)

        # Apply binary mask to the original image to remove colors that are not black
        # result = cv2.bitwise_and(image, image, mask=binary)
        # cv2.imshow("",binary)
        # # Convert the result to grayscale
        # gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # # Apply binary thresholding to create a binary image
        # _, binary_result = cv2.threshold(gray_result, 96, 255, cv2.THRESH_BINARY)

        # Define the kernel for dilation
        kernel = np.ones((2, 2), np.uint8)

        # Apply dilation to thicken the lines
        dilated = cv2.dilate(binary, kernel, iterations=2)
        # cv2.imshow("",dilated)
        plt.imshow(dilated)
        plt.show()
        return dilated

    def extractPoints(self):
        params = cv2.SimpleBlobDetector_Params()

        # Adjust the parameters according to your image and requirements
        params.minThreshold = 0
        params.maxThreshold = 32
        params.filterByArea = True
        params.minArea = 0.1  # Minimum area for a circular blob with a radius of 2 pixels
        params.filterByCircularity = False
        # params.minCircularity = 0  # Minimum circularity value for blobs
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minDistBetweenBlobs = 0.01  # Minimum distance between blobs, set to 0 for a dense point set

        # Create the blob detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs in the binary image
        keypoints = detector.detect(self.dilated_canvas)

        # Extract the coordinates of the keypoints
        coordinates = [kp.pt for kp in keypoints]

        # Sort the points based on 'x' coordinate
        coordinates = sorted(coordinates, key=lambda p: p[0])
        max_X = (coordinates[len(coordinates)-1][0] - coordinates[0][0])  

        maxy = max([coord[1] for coord in coordinates])
        maxx = max([coord[0] for coord in coordinates])
        for i in coordinates:
            x = rescale(i[0], [0, max_X], self.x_range)
            y = rescale(i[1], [0, maxy], self.y_range)
            self.points.append([x, (y*-1)+self.y_range[1]])

        self.points = rescale_points(self.points, self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1])

        myx = [coord[0] for coord in self.points]
        myy = [coord[1] for coord in self.points]

        # if self.show:
        plt.plot(myx, myy, linestyle="-", color="blue")
        plt.plot(myx, myy, marker=".", linestyle="", color="black")
        plt.show()