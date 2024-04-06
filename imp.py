import cv2
import numpy as np

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

image_path = 'image.jpg'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

normalized_image = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

gray_image_resized = resize_image(gray_image, 50)
normalized_image_resized = resize_image(normalized_image, 50)

cv2.imshow('Original Image', gray_image_resized)
cv2.imshow('Sharpened Image', normalized_image_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
