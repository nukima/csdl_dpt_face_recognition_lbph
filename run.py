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

def get_top_similar_images(image_path, method='euclidean', top=10):
    # Extract feature from uploaded image
    lbp_feature = extract_lbp_feature(image_path, haar_cascade, lbp)
    histogram = extract_histogram(lbp_feature, 4, 4)
    compare_result = []
    for image_name, image_histogram in histograms.items():
        similarity = compare_histograms(image_histogram, histogram, method)
        compare_result.append((image_name, similarity))
    # get top 10 similar images (sorted by similarity ascending)
    compare_result = sorted(compare_result, key=lambda x: x[1])
    top_similar_paths = []
    for i in range(top):
        top_similar_paths.append(image_base_path + compare_result[i][0] + '.png')

    return top_similar_paths

# Function to handle search button click
def search_images(search_mode):
    print("Search mode: " + search_mode)
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((100, 100))
        img = ImageTk.PhotoImage(img)
        uploaded_image.configure(image=img)
        uploaded_image.image = img
    

    # Display top 10 similar face images
    top_similar_paths = get_top_similar_images(file_path, search_mode, 10)

    # Clear previous search results
    for label in result_labels:
        label.configure(image="")
        label.image = ""

   # Display top 10 similar images in two rows
    for i, path in enumerate(top_similar_paths):
        img = Image.open(path)
        img.thumbnail((200, 200))  # Resize the image to fit within the window
        img = ImageTk.PhotoImage(img)
        result_labels[i].configure(image=img, text="Top " + str(i + 1))
        result_labels[i].image = img
        result_labels[i].update()

def quit_program():
    window.destroy()

if __name__ == "__main__":

    window = tk.Tk()
    window.title("Face Image Search")
    window.configure(background='white')
    window.geometry("1200x600")

    # Create left layout for buttons
    left_frame = tk.Frame(window, bg='white')
    left_frame.pack(side=tk.LEFT, padx=20, pady=20)
    left_frame.place(width=200, height=600)

    search_mode_label = tk.Label(left_frame, text="Histogram compare method:", bg='white')
    search_mode_label.pack(padx=10, pady=5)

    search_mode = tk.StringVar()
    search_mode.set("euclidean")  # Default search mode

    mode1_button = tk.Radiobutton(left_frame, text="Euclidean", variable=search_mode, value="euclidean", font=("Hack", 12))
    mode1_button.pack(padx=10, pady=5)
    # mode2_button = tk.Radiobutton(left_frame, text="Normalized Euclidean", variable=search_mode, value="normalized_euclidean", font=("Hack", 12))
    # mode2_button.pack(padx=10, pady=5)
    mode3_button = tk.Radiobutton(left_frame, text="Absolute Value", variable=search_mode, value="absolute_value", font=("Hack", 12))
    mode3_button.pack(padx=10, pady=5)
    # mode4_button = tk.Radiobutton(left_frame, text="Chi Square", variable=search_mode, value="chi_square")
    # mode4_button.pack(padx=10, pady=5)

    search_button = tk.Button(left_frame, text="Search", command=lambda: search_images(search_mode.get()),
                                height=2, width=20, font=("Hack", 20))
    search_button.pack(padx=10, pady=10)

    quit_button = tk.Button(left_frame, text="Quit", command=quit_program,
                            height=2, width=20, font=("Hack", 20))
    quit_button.pack(padx=10, pady=10)

    # Create right layout for image display
    right_frame = tk.Frame(window, bg='white')
    right_frame.pack(side=tk.LEFT, padx=20, pady=20)
    right_frame.place(x=200, width=1000, height=600)

    # Upper row - uploaded image
    uploaded_image = tk.Label(right_frame, bg='white')
    uploaded_image.pack(padx=10, pady=10)

    # Lower row - search results: display in two rows
    result_frame = tk.Frame(right_frame, bg='white')
    result_frame.pack(padx=10, pady=10)
    result_labels = []
    for i in range(10):
        label = tk.Label(result_frame, bg='white', text="Top " + str(i + 1), compound=tk.TOP)
        label.grid(row=i // 5, column=i % 5, padx=10, pady=10)
        result_labels.append(label)

    

    window.mainloop()
