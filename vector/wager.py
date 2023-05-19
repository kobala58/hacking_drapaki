from typing import Dict, List
import random
import os
import pytesseract
from PIL import Image
import json

from Levenshtein import distance as lev

os.environ['TESSDATA_PREFIX'] = 'C:\\Program Files (x86)\\Tesseract-OCR\\tessdata'
tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'


def normalizator():
    base = "./odciete/"
    for file in os.listdir(base):
        print(file)
        with open(f"{base}{file}", "r") as j_file:
            data = json.loads(j_file.read())
        summ = 0
        for key, val in data.items():
            summ += val

        for key, val in data.items():
            data[key] = (val / summ).__round__(6)

        for key, val in data.items():
            print(f"{key}: [{val}]")

        with open(f"{base}norm_{file}", "w") as j_file:
            test = json.dumps(data, indent=4)
            j_file.write(test)


def closest_class(data: list, norms: List[Dict]) -> int:
    """
    :param data: list of string with message
    :param norms: List of Dicts from normalized folder
    :return: Number of assigned class
    """
    res = dict()

    for norm in norms:

        res[norm["name"]] = 0
        for x in data:
            for y in norm["words"].keys():
                if lev(x, y) < 3:
                    res[norm["name"]] += norm["words"][y]

    # for key, val in res.items():
    #     print(f"{key}: {val}")

    new = sorted(res.items(), key=lambda z: z[1], reverse=True)[0]

    return new[0]


def __random_image():
    threshold = 150
    dirname = "../datasets/train_set/pit37_v1"
    file = random.choice(os.listdir(dirname))

    img = Image.open(dirname + "/" + file).convert("LA")
    img = img.point(lambda p: 255 if p > threshold else 0)

    ret = pytesseract.image_to_string(img)

    data = [x for x in ret.strip().lower().split() if len(x) > 3]
    print(data)
    return data


def load_norms(path) -> list:
    data = []
    for x in os.listdir(path):
        with open(f"{path}/{x}") as file:
            j_data = json.loads(file.read())
            data.append(j_data)
    return data


def main():
    norms = load_norms('norms')
    txt = __random_image()
    print(closest_class(txt, norms))


if __name__ == '__main__':
    main()
