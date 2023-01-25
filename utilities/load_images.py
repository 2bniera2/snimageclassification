import cv2
import os

# load images from a path into a list
def load_images(path):  
    X = []
    labels = []

    for class_name in os.listdir(path):
        file_list = os.listdir(f"{path}/{class_name}")

        for index, file in enumerate(file_list):
            # open image and convert to RGB
            im = cv2.imread(f"{path}/{class_name}/{file}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 

            X.append(im)

            labels.append(class_name)

    return X, labels