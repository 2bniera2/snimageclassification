import cv2
import os




# append paths to images into a list
def load_images(path):  
    X = []
    labels = []

    for class_name in os.listdir(path):
        file_list = os.listdir(f"{path}/{class_name}")

        for file in file_list:
            X.append(f"{path}/{class_name}/{file}")
            labels.append(class_name)

    return X, labels