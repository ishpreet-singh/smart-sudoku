import numpy as np
import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import sys
from controller.sudoku_controller import SudokuController

class ImageController:

    def __init__(self):
        pass
    
    def blurring(self, img, kernel_size):
        return cv2.GaussianBlur(img, kernel_size, 0)

    def adaptive_thresholding(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 11, 2)

    def invertion(self,img):
        return cv2.bitwise_not(img,img)

    def dilation(self, img, kernel):
        return cv2.dilate(img, kernel)

    def erosion(self, img, kernel):
        return cv2.erode(img, kernel, iterations=1) 

    def preprocess_image(self, image, skip_dilate=False):
        kernel_size = (9,9)
        processed_image = self.blurring(image.copy(), kernel_size)
        processed_image = self.adaptive_thresholding(processed_image)
        processed_image = self.invertion(processed_image)
        if not skip_dilate:
            kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
            processed_image = self.dilation(processed_image,kernel)
        return processed_image

    def find_corners_of_sudoku(self, image):
        contours, h = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
        largest_image = contours[0] 
        
        ## [
        #  [[798 180]]
        #  [[797 181]]
        #  [[784 181]]
        #   ..
        #  [[809 187]]
        #  [[808 186]]
        #  [[808 180]]
        ## ]

        # Bottom-right point has the largest (x + y) value
        # Top-left has point smallest (x + y) value
        # Bottom-left point has smallest (x - y) value
        # Top-right point has largest (x - y) value
        bottom_right, _ = max(enumerate([point[0][0] + point[0][1] for point in largest_image]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([point[0][0] + point[0][1] for point in largest_image]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([point[0][0] - point[0][1] for point in largest_image]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([point[0][0] - point[0][1] for point in largest_image]), key=operator.itemgetter(1))

        # Return an array of all 4 points using the indices
        # Each point is in its own array of one coordinate
        return [largest_image[top_left][0], largest_image[top_right][0], largest_image[bottom_right][0], largest_image[bottom_left][0]]


    def euclidean_between(self, point1, point2):
        a = point2[0] - point1[0]
        b = point2[1] - point1[1]
        return np.sqrt((a ** 2) + (b ** 2))

    def crop_image(self, image, corners):
        # Rectangle described by top left, top right, bottom right and bottom left points
        top_left = corners[0]
        top_right = corners[1]
        bottom_right = corners[2]
        bottom_left = corners[3]

        # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

        # Get the longest side in the rectangle
        side = max([
            self.euclidean_between(bottom_right, top_right),
            self.euclidean_between(top_left, bottom_left),
            self.euclidean_between(bottom_right, bottom_left),
            self.euclidean_between(top_left, top_right)
        ])

        # Describe a square with side of the calculated length, this is the new perspective we want to warp to
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

        # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
        m = cv2.getPerspectiveTransform(src, dst)

        # Performs the transformation on the original image
        return cv2.warpPerspective(image, m, (int(side), int(side)))

    def scale_and_centre(self, img, size, margin=0, background=0):
        """Scales and centres an image onto a new background square."""
        h, w = img.shape[:2]

        def centre_pad(length):
            """Handles centering for a given length that may be odd or even."""
            if length % 2 == 0:
                side1 = int((size - length) / 2)
                side2 = side1
            else:
                side1 = int((size - length) / 2)
                side2 = side1 + 1
            return side1, side2

        def scale(r, x):
            return int(r * x)

        if h > w:
            t_pad = int(margin / 2)
            b_pad = t_pad
            ratio = (size - margin) / h
            w, h = scale(ratio, w), scale(ratio, h)
            l_pad, r_pad = centre_pad(w)
        else:
            l_pad = int(margin / 2)
            r_pad = l_pad
            ratio = (size - margin) / w
            w, h = scale(ratio, w), scale(ratio, h)
            t_pad, b_pad = centre_pad(h)

        img = cv2.resize(img, (w, h))
        img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
        return cv2.resize(img, (size, size))


    def find_largest_feature(self, image, scan_tl=None, scan_br=None):
        """
        Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
        connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
        """
        img = image.copy() # Copy the image, leaving the original untouched
        #show_image(img,"sd")
        height, width = img.shape[:2]

        max_area = 0
        seed_point = (None, None)

        if scan_tl is None:
            scan_tl = [0, 0]

        if scan_br is None:
            scan_br = [width, height]

        # Loop through the image
        for x in range(scan_tl[0], scan_br[0]):
            for y in range(scan_tl[1], scan_br[1]):
                # Only operate on light or white squares
                if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                    area = cv2.floodFill(img, None, (x, y), 64)
                    if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                        max_area = area[0]
                        seed_point = (x, y)

        # Colour everything grey (compensates for features outside of our middle scanning range
        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 255 and x < width and y < height:
                    cv2.floodFill(img, None, (x, y), 64)

        mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

        # Highlight the main feature
        if all([p is not None for p in seed_point]):
            cv2.floodFill(img, mask, seed_point, 255)

        top, bottom, left, right = height, 0, width, 0

        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                    cv2.floodFill(img, mask, (x, y), 0)

                # Find the bounding parameters
                if img.item(y, x) == 255:
                    top = y if y < top else top
                    bottom = y if y > bottom else bottom
                    left = x if x < left else left
                    right = x if x > right else right

        bbox = [[left, top], [right, bottom]]
        return img, np.array(bbox, dtype='float32'), seed_point


    def get_digit_square(self, image, square):
        return image[int(square[0][1]):int(square[1][1]), int(square[0][0]):int(square[1][0])]

    def extract_digit(self, img, rect, size):
        """Extracts a digit (if one exists) from a Sudoku square."""

        digit = self.get_digit_square(img, rect)  # Get the digit box from the whole square
        #show_image(digit,"d")
        # Use fill feature finding to get the largest feature in middle of the box
        # Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
        h, w = digit.shape[:2]
        margin = int(np.mean([h, w]) / 2.5)
        _, bbox, seed = self.find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
        digit = self.get_digit_square(digit, bbox)

        # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]

        # Ignore any small bounding boxes
        if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
            return self.scale_and_centre(digit, size, 4)
        else:
            return np.zeros((size, size), np.uint8)

    def get_squares_coordinates(self, cropped_image, number_of_squares):
        sudoku_squares = []
        size_one_square = cropped_image.shape[:1][0] / number_of_squares
        # print(size_one_square)

        for j in range(number_of_squares):
            for i in range(number_of_squares):
                point1_of_square = ( i * size_one_square, j * size_one_square )
                point2_of_square = ( (i+1) * size_one_square , (j+1) * size_one_square )
                sudoku_squares.append((point1_of_square, point2_of_square))
        # print(sudoku_squares)
        return sudoku_squares

    def get_digits(self, cropped_image, sudoku_squares, size_of_digit_image):
        sudoku_digits = []
        img = self.preprocess_image(cropped_image.copy(), skip_dilate=True)
        # show_image(img,"Dilated")
        for square in sudoku_squares:
            sudoku_digits.append(self.extract_digit(img, square, size_of_digit_image))
            # sudoku_dig= self.extract_digit(img, square, size_of_digit_image)
            # im = np.asarray(sudoku_dig)
            # show_image(sudoku_digits.reshape(28,28),"ad")
        return sudoku_digits

    def show_digits(self, digits, colour=255):
        """Shows list of 81 extracted digits in a grid format"""
        rows = []
        with_border = [cv2.copyMakeBorder(digit_img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for digit_img in digits]
        for i in range(9):
            row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
            rows.append(row)
        digit_img = self.show_image(np.concatenate(rows))
        return digit_img

    def show_image(self,img):
        """Shows an image until any key is pressed"""
        # print(type(img))
        # print(img.shape)
        # cv2.imshow('image', img)  # Display the image
        # #cv2.imwrite('images/gau_sudoku3.jpg', img)
        # cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
        # cv2.destroyAllWindows()  # Close all windows
        return img
    def output(self, a):
        sys.stdout.write(str(a))

    def display_sudoku(self, sudoku):
        for i in range(9):
            for j in range(9):
                cell = sudoku[i][j]
                if cell == 0 or isinstance(cell, set):
                    self.output('.')
                else:
                    self.output(cell)
                if (j + 1) % 3 == 0 and j < 8:
                    self.output(' |')

                if j != 8:
                    self.output('  ')
            self.output('\n')
            if (i + 1) % 3 == 0 and i < 8:
                self.output("--------+----------+---------\n")

    def identify_number(self, image, loaded_model):
        image_resize = cv2.resize(image, (28,28))    # For plt.imshow
        # image_resize_2 = image_resize.reshape(1,28,28,1) 
        image_resize_2 = image_resize.reshape(1,1, 28,28)
        loaded_model_pred = loaded_model.predict_classes(image_resize_2 , verbose = 0)
        return loaded_model_pred[0]

    def extract_number(self, sudoku, loaded_model):
        sudoku = cv2.resize(sudoku, (450,450))
        # split sudoku
        grid = np.zeros([9,9])
        for i in range(9):
            for j in range(9):
                image = sudoku[i*50:(i+1)*50,j*50:(j+1)*50]
                if image.sum() > 25000:
                    grid[i][j] = self.identify_number(image, loaded_model)
                    # print("Number: ",grid[i][j])
                else:
                    grid[i][j] = 0
        return grid.astype(int)

    def controller(self, path):
        '''
        Step 1: Preprocess image -> blur, threshold, invertion and dilation
        '''
        original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        processed = self.preprocess_image(original, False)
        show_image(original,'Original Image')
        '''
        Step 2: Get Sudoku Square -> Find the largest contour (sudoku square) and Get the cordinates of largest contour.
        '''
        corners = self.find_corners_of_sudoku(processed)
        '''
        Step 3: Crop Image -> using wrap
        '''  
        cropped_image= self.crop_image(original,corners)
        show_image(cropped_image,'Cropped Image')
        # cropped_img = cv2.resize(cropped_image, (500, 500))
        '''
        Step 4: Get Squares coordinates from the cropped image of sudoku
        '''  
        number_of_squares = 9
        sudoku_squares = self.get_squares_coordinates(cropped_image,number_of_squares)

        '''
        Step 5: Get digits from each square of sudoku
        '''   
        size_of_digit_image = 28 # as per MNIST dataset
        sudoku_digits = self.get_digits(cropped_image, sudoku_squares, size_of_digit_image)   
        #print(sudoku_digits)
        '''
        Step 6: Show Digits
        '''
        final_image = self.show_digits(sudoku_digits)
        show_image(final_image,"Final Image")

        return final_image
        

def show_image(image, title):
    plt.imshow(image,'gray')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    ic = ImageController() 

    # Load the saved model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    
    path = '../dataset/Sample1.jpg'
    image = ic.controller(path)

    grid = ic.extract_number(image, loaded_model)
    ic.display_sudoku(grid.tolist())

    sc = SudokuController(grid)

    solution = sc.sudoku_solver(grid)
    print('Solution:')
    ic.display_sudoku(solution.tolist())

