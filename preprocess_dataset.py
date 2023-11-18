# preprocess_dataset.py

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

def resize_images(folder_path, target_size=(64, 64)):
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, target_size)
        cv2.imwrite(img_path, resized_img)

def normalize_images(folder_path):
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        normalized_img = img / 255.0
        cv2.imwrite(img_path, normalized_img)

def split_dataset(source_folder, train_folder, test_folder, test_size=0.2):
    for gesture_folder in os.listdir(source_folder):
        gesture_path = os.path.join(source_folder, gesture_folder)
        train_path = os.path.join(train_folder, gesture_folder)
        test_path = os.path.join(test_folder, gesture_folder)

        # Create train and test folders
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Split the images into train and test sets
        images = os.listdir(gesture_path)
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)

        # Move images to the corresponding folders
        for img in train_images:
            shutil.copy(os.path.join(gesture_path, img), os.path.join(train_path, img))

        for img in test_images:
            shutil.copy(os.path.join(gesture_path, img), os.path.join(test_path, img))

# Example usage
resize_images('dataset/rock')
resize_images('dataset/paper')
resize_images('dataset/scissors')

normalize_images('dataset/rock')
normalize_images('dataset/paper')
normalize_images('dataset/scissors')

split_dataset('dataset', 'train_dataset', 'test_dataset')
