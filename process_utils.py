import os
import math
import cv2
import numpy as np
from lbp import LBP

def extract_histogram(pixels, grid_x, grid_y):
    hist = []

    # Check the pixels matrix
    if len(pixels) == 0:
        raise ValueError("The pixels list passed to the calculate function is empty")

    # Get the matrix dimensions
    rows = len(pixels)
    cols = len(pixels[0])

    # Check the grid (X and Y)
    if grid_x <= 0 or grid_x >= cols:
        raise ValueError("Invalid grid X passed to the calculate function")
    if grid_y <= 0 or grid_y >= rows:
        raise ValueError("Invalid grid Y passed to the calculate function")

    # Get the size (width and height) of each region
    grid_width = cols // grid_x
    grid_height = rows // grid_y

    # Calculate the histogram of each grid
    for g_x in range(grid_x):
        for g_y in range(grid_y):
            # Create a list with empty 256 positions
            region_histogram = [0] * 256

            # Define the start and end positions for the following loop
            start_pos_x = g_x * grid_width
            start_pos_y = g_y * grid_height
            end_pos_x = (g_x + 1) * grid_width
            end_pos_y = (g_y + 1) * grid_height

            # Make sure that no pixel has been left at the end
            if g_x == grid_x - 1:
                end_pos_x = cols
            if g_y == grid_y - 1:
                end_pos_y = rows

            # Create the histogram for the current region
            for x in range(start_pos_x, end_pos_x):
                for y in range(start_pos_y, end_pos_y):
                    # Make sure we are trying to access a valid position
                    if x < len(pixels) and y < len(pixels[x]):
                        pixel_value = pixels[x][y]
                        if pixel_value < len(region_histogram):
                            region_histogram[int(pixel_value)] += 1

            # Concatenate two lists
            hist.extend(region_histogram)

    return hist

def extract_lbp_feature(image_path, haar_cascade, lbp):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    if len(faces_rect) == 0:
        raise ValueError("No face detected")
    (x, y, w, h) = faces_rect[0]
    face = gray_img[y:y + w, x:x + h]
    face = cv2.resize(face, (256, 256))
    # face = cv2.equalizeHist(face)
    lbp_feature = lbp(face)

    return lbp_feature

def write_feature_to_file(feature, image_name, feature_base_path):
    with open(feature_base_path + image_name + '.txt', 'w') as f:
        f.write(str(feature))
        f.close()


def euclidean_distance(hist1, hist2):
    if len(hist1) != len(hist2):
        raise ValueError("Histogram sizes do not match")

    sum = 0.0
    for index in range(len(hist1)):
        sum += math.pow(hist1[index] - hist2[index], 2)

    return math.sqrt(sum)

def normalized_euclidean_distance(hist1, hist2):
    # Check the histogram sizes
    if len(hist1) != len(hist2):
        raise ValueError("Histogram sizes do not match")

    sum = 0.0
    n = float(len(hist1))
    for index in range(len(hist1)):
        sum += math.pow(hist1[index] - hist2[index], 2) / n

    return math.sqrt(sum)

def absolute_value(hist1, hist2):
    # Check the histogram sizes
    if len(hist1) != len(hist2):
        raise ValueError("Histogram sizes do not match")

    sum = 0.0
    for index in range(len(hist1)):
        sum += math.fabs(hist1[index] - hist2[index])

    return sum

def chi_square(hist1, hist2):
    # Check the histogram sizes
    if (len(hist1) != len(hist2)) or (len(hist1) == 0) or (len(hist2) == 0):
        raise ValueError("Histogram sizes do not match")

    sum = 0.0
    for index in range(len(hist1)):
        numerator = math.pow(hist1[index] - hist2[index], 2)
        denominator = hist1[index]
        sum += numerator / denominator

    return sum

def compare_histograms(hist1, hist2, method):
    # Check the histogram sizes
    if len(hist1) != len(hist2):
        raise ValueError("Histogram sizes do not match")

    if method == 'euclidean':
        return euclidean_distance(hist1, hist2)
    elif method == 'normalized_euclidean':
        return normalized_euclidean_distance(hist1, hist2)
    elif method == 'absolute_value':
        return absolute_value(hist1, hist2)
    elif method == 'chi_square':
        return chi_square(hist1, hist2)
    else:
        raise ValueError("Invalid comparison method")
