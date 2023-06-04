import os
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np

from process_utils import *

lbp = LBP(radius=1, npoints=8, counter_clockwise=True, interpolation="bilinear")
haar_cascade = cv2.CascadeClassifier('resource/haarcascade_frontalface_default.xml')
image_base_path = '/home/ubuntu/Documents/csdl_dpt/src/background-remover/'
feature_base_path = '/home/ubuntu/Documents/csdl_dpt/src/histograms/'
histograms = {}
for histogram_file in os.listdir(feature_base_path):
    image_name = histogram_file.split('.')[0]
    with open(feature_base_path + histogram_file, 'r') as f:
        histogram = f.read()
        histogram = histogram.replace('[', '')
        histogram = histogram.replace(']', '')
        histogram = histogram.split(',')
        histogram = [float(x) for x in histogram]
        histograms[image_name] = histogram
        f.close()

# find all duplicate images
duplicate_images = {}
# add flag to know if image is already checked
checked_images = {}
for image_name, image_histogram in histograms.items():
    if image_name not in checked_images:
        checked_images[image_name] = True
        for another_image_name, another_image_histogram in histograms.items():
            if another_image_name not in checked_images:
                similarity = compare_histograms(image_histogram, another_image_histogram, 'euclidean')
                if similarity < 0.1:
                    if image_name not in duplicate_images:
                        duplicate_images[image_name] = []
                    duplicate_images[image_name].append(another_image_name)
                    checked_images[another_image_name] = True

print(duplicate_images)