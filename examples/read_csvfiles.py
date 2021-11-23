import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
import csv
# List the images' paths (PNG files must have transparent backgrounds)
mask_filenames = ['image/ryan_transparent.png', 'image/batman_2.png', 'image/iron_man_2.png', 'image/none.png']

# Pixel coordinates must match from the images to each landmark
csv_filenames = ['image/ryan_transparent.csv', 'image/batman_2.csv', 'image/iron_man_2.csv']


def readCSV(file):
    landmarks = {}
    ids = []
    coordinates = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (line_count != 0):
                landmarks[line_count] = {'id': int(row[0]),
                                         'x': int(row[1]),
                                         'y': int(row[2])}
                ids.append(int(row[0]))
                coordinates.append([int(row[1]), int(row[2])])

            line_count += 1

    return landmarks, ids, coordinates





# csv_filename = csv_filenames[0 if selected == 3 else selected]
csv_filename = csv_filenames[0]

# img_filename = mask_filenames[selected]
landmarks, ids, mask_coordinates = readCSV(csv_filename)


