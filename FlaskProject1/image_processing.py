#1-adsoyad:amir elahmed
#1-öğrno:2112721307
#2-adsoyad:mohamad alkassem
#2-öğrno:2212721320


import cv2
import numpy as np

def process_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    edges = cv2.Canny(blurred, 50, 150)


    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)


    corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    corners = cv2.dilate(corners, None)


    corner_image = image.copy()
    corner_image[corners > 0.01 * corners.max()] = [0, 0, 255]


    contour_image = image.copy()
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)


    combined = cv2.addWeighted(contour_image, 0.7, corner_image, 0.3, 0)
    return combined

def apply_threshold(image, threshold_value=128):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
