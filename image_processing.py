
import os
import cv2
import numpy as np
import operator
import matplotlib.pyplot as plt
from digit_recoganizer import DigitRecoganizer


class ImageProcessing():

    def __init__(self, image_path):
        self.image = cv2.imread(img_path, 0)
        self.original_image = self.image.copy()
        # self.kernel = np.ones((3,3),np.uint8)
        self.kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

    def show(self):
        plt.imshow(self.image, 'gray')
        plt.show()

    def blurr(self):
        self.image = cv2.GaussianBlur(self.image, (9, 9), 0)

    def invert(self):
        self.image = cv2.bitwise_not(self.image)

    def erosion(self):
        self.image = cv2.erode(self.image, self.kernel)

    def dilation(self):
        self.image = cv2.dilate(self.image, self.kernel)

    def adaptive_thresholding(self):
        self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    def preprocess(self):
        self.blurr()
        self.adaptive_thresholding()
        self.invert()
        # self.dilation()

    def find_corners_of_largest_polygon(self):
        """Finds the 4 extreme corners of the largest contour in the image."""
        contours, h = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        # print("Countours: ", contours)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
        polygon = contours[0]  # Largest image

        # cv2.drawContours(self.image, contours, -1, (0,0,0), 5)
        # cv2.imshow('Countours', self.image)
        # cv2.waitKey(5000)

        # Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
        # Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

        # Bottom-right point has the largest (x + y) value
        # Top-left has point smallest (x + y) value
        # Bottom-left point has smallest (x - y) value
        # Top-right point has largest (x - y) value
        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

        # Return an array of all 4 points using the indices
        # Each point is in its own array of one coordinate
        return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

    def distance_between(self, p1, p2):
        """Returns the scalar distance between two points"""
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        return np.sqrt((a ** 2) + (b ** 2))

    def crop_and_warp(self, corners):
        """Crops and warps a rectangular section from an image into a square of similar size."""

        # Rectangle described by top left, top right, bottom right and bottom left points
        top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]

        # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

        print("top_left: ", top_left)
        print("top_right: ", top_right)
        print("bottom_right: ", bottom_right)
        print("bottom_left: ", bottom_left)

        # Get the longest side in the rectangle
        side = max([
            self.distance_between(bottom_right, top_right),
            self.distance_between(top_left, bottom_left),
            self.distance_between(bottom_right, bottom_left),
            self.distance_between(top_left, top_right)
        ])

        # Describe a square with side of the calculated length, this is the new perspective we want to warp to
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

        # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
        m = cv2.getPerspectiveTransform(src, dst)

        # Performs the transformation on the original image
        return cv2.warpPerspective(self.image, m, (int(side), int(side)))


    def flood_fill(self):
        height, width = np.shape(self.image)
        img = self.image
        max_area = -1
        max_point = (0, 0)

        for row in range(width):
            for col in range(height):
                if img[col][row] >= 128:
                    area = cv2.floodFill(img, None, (row, col), 64)[0]
                    if area > max_area:
                        max_point = (row, col)
                        max_area = area

        # Floodfill the biggest blob with white (Our sudoku board's outer grid)
        cv2.floodFill(img, None, max_point, (255, 255, 255))

        for row in range(width):
            for col in range(height):
                if img[col][row] == 64 and row != max_point[0] and col != max_point[1]:
                    cv2.floodFill(img, None, (row, col), 0)


    def grid_extraction(self):
        self.image = cv2.bitwise_not(cv2.adaptiveThreshold(
            self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1))

        grid = np.copy(self.image)
        edge = np.shape(grid)[0]
        celledge = edge // 9

        #Creating a vector of size 81 of all the cell images
        tempgrid = []
        for i in range(celledge, edge+1, celledge):
            for j in range(celledge, edge+1, celledge):
                rows = grid[i-celledge:i]
                tempgrid.append([rows[k][j-celledge:j]
                                 for k in range(len(rows))])

        #Creating the 9X9 grid of images
        finalgrid = []
        for i in range(0, len(tempgrid)-8, 9):
            finalgrid.append(tempgrid[i:i+9])

        #Converting all the cell images to np.array
        for i in range(9):
            for j in range(9):
                finalgrid[i][j] = np.array(finalgrid[i][j])

        return finalgrid


if __name__ == "__main__":
    
    img_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "dataset/sample1.jpg")
    image = ImageProcessing(img_path)

    image.preprocess()
    corners = image.find_corners_of_largest_polygon()
    cropped_image = image.crop_and_warp(corners)

    cropped_img = cv2.resize(image.image, (500, 500))              # Resize image
    cv2.imshow('cropped', cropped_image)
    cv2.waitKey(5000)

    # dg = DigitRecoganizer()
    # dg.load_data()
    # dg.normalize()
    # dg.cnn_model()
    # dg.train()

    # image.show()
    # image.show()
    # image.show()

    # # image.flood_fill()
    # # image.erosion()
    # # image.show()
    # # image.hough_transform()
    # # image.show()
    # # image.grid_extraction()
    # # image.show()
    # # image.grid_detection()

    # scanned_images = image.grid_extraction()
    # # for i in range(9):
    # #     for j in range(9):
    # cell_image = cv2.resize(scanned_images[0][0], (28, 28))
    
    # recoganised_digit = dg.predict(cell_image)
    # print(recoganised_digit, " ")
    # print("\n")
            
        
        
    