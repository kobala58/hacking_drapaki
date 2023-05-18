import os
# import shutil

import pytesseract
# import re
from PIL import Image
# from skimage.morphology import opening
# from skimage.morphology import disk

import tensorflow as tf

# import tensorflow_hub as hub
# import tensorflow_text as text
# from tensorflow import keras

# import matplotlib.pyplot as plt

# os.putenv('TESSDATA_PREFIX', 'C:/Program Files (x86)/Tesseract-OCR/tessdata/osd.traineddata')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


def main(path: str = './datasets/train_set/pit37_v1'):
    keywords = str("DANE").lower()
    # black = (0, 0, 0)
    # white = (255, 255, 255)
    # threshold = (160, 160, 160)

    hit = 0
    total = 0
    error = 0
    dir_len = len(os.listdir(path))
    threshold = 160
    for file in os.listdir(path):

        # Open input image in grayscale mode and get its pixels.
        # img = Image.open(dir + "/" + file).convert("LA")
        # pixels = img.getdata()

        # newPixels = []

        # Compare each pixel
        with Image.open(f'{path}/{file}') as img:
            # img_gray = img.convert('L')
            img_bin = img.point(lambda x: 255 if x > threshold else 0, mode='1')
            # img_resized = img_bin.resize((256, 256))

            # Create and save new image.
            # newImg = Image.new("RGB", img.size)
            # newImg.putdata(newPixels)
            # newImg.save("newImage3.jpg")

            # print(pytesseract.image_to_string(Image.open(file)))
            # print("---------------------------------------------------------------------------------------")
            ret = pytesseract.image_to_string(img_bin)
            if keywords in ret.lower():
                hit += 1
                total += 1
            else:
                error += 1
                total += 1

        print(f"{total}/{dir_len}, HITS% {hit / total}")

    print(f"Final % : {(hit * 100 / total).__round__(2)}% ({hit})")


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    main()
