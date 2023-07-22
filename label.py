import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract as pts
from extractor import Canvas
# import easyocr
from constants import *
"""
THIS FILE IS FOR EXTRACTING THE LABEL
"""

class LabelExtractor():
    def __init__(self, image: Canvas, default_max=100, default_min=0):
        self.labels = None
        self.canvas = image
        self.default_max = default_max
        self.default_min = default_min
        self.max_x = self.default_max
        self.max_y = self.default_max
        self.min_x = self.default_min
        self.min_y = self.default_min
        self.maxyimg = None
        self.minyimg = None
        self.maxximg = None
        self.minximg = None
        self.custom_config = r'--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'

    def extractLabel(self):
        pts.pytesseract.tesseract_cmd = TESSERACT
        maxy_x, maxy_y, maxy_w, maxy_h = (0+int(self.canvas.width*0.03), self.canvas.bb_y, self.canvas.bb_x-6, int(self.canvas.height*.185))
        miny_x, miny_y, miny_w, miny_h = (0+int(self.canvas.width*0.03), self.canvas.bb_h-int(self.canvas.height*.075), self.canvas.bb_x-6, self.canvas.bb_h-int(self.canvas.height*.01))
        maxx_x, maxx_y, maxx_w, maxx_h = (self.canvas.bb_w-int(self.canvas.width*.065), self.canvas.bb_h+6, self.canvas.bb_w, self.canvas.height-int(self.canvas.height*.025))
        minx_x, minx_y, minx_w, minx_h = (self.canvas.bb_x, self.canvas.bb_h+6, self.canvas.bb_x+int(self.canvas.width*.075), self.canvas.height-int(self.canvas.height*.025))

        self.maxyimg = self.canvas.image[maxy_y:maxy_h, maxy_x:maxy_w]
        self.minyimg = self.canvas.image[miny_y:miny_h, miny_x:miny_w]
        self.maxximg = self.canvas.image[maxx_y:maxx_h, maxx_x:maxx_w]
        self.minximg = self.canvas.image[minx_y:minx_h, minx_x:minx_w]

        self.maxyimg = cv2.cvtColor(self.maxyimg, cv2.COLOR_BGRA2GRAY)
        self.minyimg = cv2.cvtColor(self.minyimg, cv2.COLOR_BGRA2GRAY)
        self.maxximg = cv2.cvtColor(self.maxximg, cv2.COLOR_BGRA2GRAY)
        self.minximg = cv2.cvtColor(self.minximg, cv2.COLOR_BGRA2GRAY)

        # Sharpening
        # kernel = np.array([[-1,-1,-1], [-1, 8.5,-1], [-1,-1,-1]])
        # self.maxyimg = cv2.filter2D(self.maxyimg, -1, kernel)
        # self.minyimg = cv2.filter2D(self.minyimg, -1, kernel)
        # self.maxximg = cv2.filter2D(self.maxximg, -1, kernel)
        # self.minximg = cv2.filter2D(self.minximg, -1, kernel)

        # cv2.imshow("", self.minyimg)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # reader = easyocr.Reader(['en'])
        # self.max_y = reader.readtext(self.maxyimg)
        # self.min_y = reader.readtext(self.minyimg)
        # self.max_x = reader.readtext(self.maxximg)
        # self.min_x = reader.readtext(self.minximg)
        # for detection in max_y:
        #     text = detection[1]
        #     confidence = detection[2]
        #     bbox = detection[0]
        # print(self.max_y[0][1])
        # print(self.min_y[0][1])
        # print(self.max_x[0][1])
        # print(self.min_x[0][1])
            # print(f"Text: {text}, Confidence: {confidence}, Bbox: {bbox}")
        self.max_y = pts.image_to_string(self.maxyimg, config=self.custom_config)
        self.min_y = pts.image_to_string(self.minyimg, config=self.custom_config)
        self.max_x = pts.image_to_string(self.maxximg, config=self.custom_config)
        self.min_x = pts.image_to_string(self.minximg, config=self.custom_config)

        try:
            self.max_y = self.default_max if self.max_y == None or self.max_y == '' else int(self.max_y)
        except Exception as e:
            self.max_y = self.default_max
        try:
            self.min_y = self.default_min if self.min_y == None or self.min_y == '' else int(self.min_y)
        except Exception as e:
            self.min_y = self.default_min
        try:
            self.max_x = self.default_max if self.max_x == None or self.max_x == '' else int(self.max_x)
        except Exception as e:
            self.max_x = self.default_max
        try:
            self.min_x = self.default_min if self.min_x == None or self.min_x == '' else int(self.min_x)
        except Exception as e:
            self.min_x = self.default_min

    def showCanvas(self):
        maxy_x, maxy_y, maxy_w, maxy_h = (0+int(self.canvas.width*0.03), self.canvas.bb_y, self.canvas.bb_x-6, int(self.canvas.height*.185))
        miny_x, miny_y, miny_w, miny_h = (0+int(self.canvas.width*0.03), self.canvas.bb_h-int(self.canvas.height*.075), self.canvas.bb_x-6, self.canvas.bb_h-int(self.canvas.height*.01))
        maxx_x, maxx_y, maxx_w, maxx_h = (self.canvas.bb_w-int(self.canvas.width*.065), self.canvas.bb_h+6, self.canvas.bb_w, self.canvas.height-int(self.canvas.height*.025))
        minx_x, minx_y, minx_w, minx_h = (self.canvas.bb_x, self.canvas.bb_h+6, self.canvas.bb_x+int(self.canvas.width*.075), self.canvas.height-int(self.canvas.height*.025))

        result = cv2.rectangle(self.canvas.image.copy(), (self.canvas.bb_x, self.canvas.bb_y), (self.canvas.bb_w, self.canvas.bb_h), (0, 255, 0), 2)
        result = cv2.rectangle(result, (maxy_x, maxy_y), (maxy_w, maxy_h), (255, 0, 0), 1)
        result = cv2.rectangle(result, (miny_x, miny_y), (miny_w, miny_h), (0, 0, 255), 1)
        result = cv2.rectangle(result, (maxx_x, maxx_y), (maxx_w, maxx_h), (255, 0, ), 1)
        result = cv2.rectangle(result, (minx_x, minx_y), (minx_w, minx_h), (0, 0, 255), 1)

        
        # Create a figure with subplots
        fig, axs = plt.subplots(3, 2)

        # Display images in subplots
        axs[0, 0].imshow(self.maxyimg)
        axs[0, 0].set_title('Max Y')
        axs[0, 0].axis("off")
        axs[0, 1].imshow(self.minyimg)
        axs[0, 1].set_title('Min Y')
        axs[0, 1].axis("off")
        axs[1, 0].imshow(self.maxximg)
        axs[1, 0].set_title('Max X')
        axs[1, 0].axis("off")
        axs[1, 1].imshow(self.minximg)
        axs[1, 1].set_title('Min X')
        axs[1, 1].axis("off")
        axs[2, 0].imshow(result)
        axs[2, 0].set_title('Result')
        axs[2, 0].axis("off")

        # Remove empty subplot
        fig.delaxes(axs[2, 1])

        # Set spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()

    def viewBoundingBoxes(self):
        ...

    def blur(self, image):
        ...

    def sharpen(self, image):
        ...
    
    def contrast(self, image):
        ...
