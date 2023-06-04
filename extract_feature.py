import os
import math
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from lbp import LBP

from process_utils import extract_lbp_feature, extract_histogram, write_feature_to_file

lbp = LBP(radius=1, npoints=8, counter_clockwise=True, interpolation="bilinear")
haar_cascade = cv2.CascadeClassifier('resource/haarcascade_frontalface_default.xml')
image_base_path = '/home/ubuntu/Documents/csdl_dpt/src/background-remover/'
feature_base_path = '/home/ubuntu/Documents/csdl_dpt/src/histograms/'
failed_images = []

for image_path in os.listdir(image_base_path):
    try:
        if image_path.endswith('.png'):
            image_name = image_path.split('.')[0]
            lbp_feature = extract_lbp_feature(image_base_path + image_path, haar_cascade, lbp)
            histogram = extract_histogram(lbp_feature, 4, 4)
            write_feature_to_file(histogram, image_name, feature_base_path)
            print('Done: ' + image_name)
    except Exception as e:
        print(e)
        failed_images.append(image_path)
        continue

print('------------------------')
print('Failed images: ')