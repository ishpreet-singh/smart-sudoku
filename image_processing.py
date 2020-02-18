
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

def blurring(image):
    Gaussian = cv2.GaussianBlur(image, (3, 3), 0)
    cv2.imshow('Gaussian Image', Gaussian) 
    return Gaussian
    # cv2.waitKey(0)  

def inversion(image):
    image = ~image   
    plt.imshow(image,'gray')
    plt.show()
    return image

def erosion(image):
    kernel = np.ones((5,5),np.uint8)
    erode = cv2.erode(image, kernel, iterations = 1)
    plt.imshow(erode,'gray')
    plt.show()
    return erode

def dilation(image):
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(image,kernel,iterations = 1)
    plt.imshow(dilate,'gray')
    plt.show()
    return dilate


def adaptive_thresholding(image):
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = ~thresh
    plt.imshow(thresh,'gray')
    # plt.imshow(~thresh,'gray')
    plt.show()
    return image

def thresholding(image):
    ret, th1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    # plt.imshow(thresh1,'gray')

    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    plt.imshow(th2,'gray')
    th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [image, th1, th2, th3]

    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    # plt.show()

if __name__ == "__main__":
    # img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/puppy.jpg")
    img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/v2_train/image1.jpg")
    # -> 0 converts it to grey scale image
    image = cv2.imread(img_path, 0)


    cv2.imshow('Image', image) 
    image = blurring(image)
    image = adaptive_thresholding(image)
    image = inversion(image)
    image = erosion(image)
    image = dilation(image)
    # cv2.waitKey(0)   

