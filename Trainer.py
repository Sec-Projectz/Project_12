import numpy as np
import os
import cv2
from PIL import Image

def get_image_data(dataset_path, dataset_path2):
    files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    files2 = [os.path.join(dataset_path2, file) for file in os.listdir(dataset_path2)]

    # print(files)

    images = []
    labels = []

    for file in files:
        try:
            img_file = Image.open(file).convert("L")
        except:
            continue

        img_data = np.array(img_file, "uint8")

        label = 2

        print("ID : ", label)

        images.append(img_data)
        labels.append(label)

    for file in files2:
        try:
            img_file = Image.open(file).convert("L")
        except:
            continue

        img_data = np.array(img_file, "uint8")

        label = 1

        print("ID : ", label)

        images.append(img_data)
        labels.append(label)

    return images, labels

def train(img_data, labels):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Training the model...")
    recognizer.train(img_data, np.array(labels))
    print("Saving the trained dataset...")
    recognizer.write('TrainedData/TrainedDataSet.yaml')
    print("Training finished...")
    print("Dataset saved under TrainedData/TrainedDataSet.yaml")

def main():
    dataset_path = "DataSet/SEAM_CARVING_ORIGINAL_Q75/Untouched"
    dataset_path2 = "DataSet/SEAM_CARVING_ORIGINAL_Q75/Seam_carved"

    print("Retreiving image data...")
    img_data, labels = get_image_data(dataset_path, dataset_path2)

    train(img_data, labels)


if __name__ == "__main__":
    main()
