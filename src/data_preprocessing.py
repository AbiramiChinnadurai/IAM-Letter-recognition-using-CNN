import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    images = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(image)
                    labels.append(label)
    
    return np.array(images), np.array(labels)

def preprocess_images(images):
    processed_images = []
    for img in images:
        img_resized = cv2.resize(img, (28, 28))
        img_normalized = img_resized / 255.0
        processed_images.append(img_normalized)
    
    return np.array(processed_images)

def split_data(images, labels, test_size=0.2, random_state=42):
    return train_test_split(images, labels, test_size=test_size, random_state=random_state)

def save_processed_data(images, labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, 'images.npy'), images)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)