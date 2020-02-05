from PIL import Image
from PIL import ImageChops
from PIL.ExifTags import TAGS
import os
import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("TrainedData/TrainedDataSet.yaml")
# recognizer.read("TrainedData/TrainedDataSet.yaml")

scale_factor = 5
temp_file = "temp.jpg"
test_file = "test.jpg"
quality_factor = 90

def metadata_check(image):
    img = Image.open(image)
    meta =  img.getexif()

    # print("meta :", meta)
    if meta:
        for (meta_tag, val) in meta.items():
            tag = TAGS.get(meta_tag, meta_tag)
            # print(tag, ":", val)
            if tag == "Software":
                print("Software Signature found...")
                print(tag,":", val)
                print("Fake Image")

                return False

    print("Metadata test passed...")
    return True

def ela_check(image):
    print("ELA test started...")

    org_img = Image.open(image)
    org_img.save(temp_file, quality = quality_factor)

    temp_img = Image.open(temp_file)

    img_diff = ImageChops.difference(org_img, temp_img)

    diff_img = img_diff.load()
    img_width, img_height = img_diff.size

    #scale the image
    for x in range(img_width):
        for y in range(img_height):
            diff_img[x,y] = tuple(pixel * scale_factor for pixel in diff_img[x,y])

    img_diff.save(test_file, quality = quality_factor)

    gray_img = Image.open(test_file).convert("L")

    # gray_img = Image.open(image).convert("L")


    gray_img_data = np.array(gray_img, "uint8")
    id, loss = recognizer.predict(gray_img_data)

    print("ID :", id)

    # if loss / 100 < 0.35:
    #     print("The image is real!")
    #     print("Confidence level :", str(loss))
    # else:
    #     print("The image is fake!")
    #     print("Confidence level :", str(loss))

    if id == 2:
        print("The image is real!")
        print("Confidence level :", str(loss))
    else:
        print("The image is fake!")
        print("Confidence level :", str(loss))


def main():
    # image = open("fake1.jpg", "rb")
    # image = open("org1.jpg", "rb")
    image = None

    try:
        path = input("Enter Image Path : ")
        image = open(path, "rb")
    except:
        print("Error in image path!")
        return

    if image is None:
        print("Error in image path!")
        return

    print("Performing metadata test...")
    if not metadata_check(image):
        return

    print("Processing ELA test...")
    ela_check(image)

if __name__ == "__main__":
    main()
