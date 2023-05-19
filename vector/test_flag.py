"""
Program dla podanego folderu ze zdjęciami zwraca listy pogrupowane po językach polskim i angielskim
Zwrócone wartości są w formatach list[dict], list[dict], gdzie dict zawiera informacje o:
    - Nazwie pliku : name
    - Listy z odczytanym tekstem ze zdjęcia : text
    Dla label = True lub type(int) także zwraca kategorię
"""

import os
import json
import typing

import pytesseract
from PIL import Image
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger
from pathlib import Path

polyglot_logger.setLevel("ERROR")


def check_lang_dir(dirpath: str, threshold=150, verbose=False, label=False) -> (
        list[dict], list[dict]):
    """
    :param dirpath: Ścieżka do zdjęć to przetworzenia
    :param threshold: Threshold od którego bit jest traktowany jako biały
    :param verbose: True jeżeli chcemy wiedzieć jaki plik aktualnie jest sprawdzany
    :param label: True jeżeli chcemy mieć label z pliku id2label przez nazwe folderu lub już numer labela
    :return:
    """
    images_en, images_pl = [], []

    if label:
        dirname = os.path.basename(os.path.normpath(dirpath))

        labels = dict(advertisement=0, budget=1, email=2, file_folder=3, form=4, handwritten=5, invoice=6, letter=7,
                      memo=8, news_article=9, pit37_v1=10, pozwolenie_uzytkowanie_obiektu_budowlanego=11,
                      presentation=12, questionnaire=13, resume=14, scientific_publication=15, scientific_report=16,
                      specification=17, umowa_na_odleglosc_odstapienie=18, umowa_o_dzielo=19,
                      umowa_sprzedazy_samochodu=20)
        label = labels[dirname]

    for _iter, file in enumerate(os.listdir(dirpath)):

        if verbose:
            print(f"{_iter + 1}/{len(os.listdir(dirpath))}", file)

        filename = os.path.join(dirpath, file)

        img = Image.open(filename).convert("LA")
        img = img.point(lambda p: 255 if p > threshold else 0)

        ocr = pytesseract.image_to_string(img)
        data = [word for word in ocr.strip().lower().split() if len(word) > 3]

        lang = check_lang(data)

        image_information = {
            "name": file,
            "text": data
        }

        if label != "False":
            image_information["label"] = label

        if lang == "pl":
            images_pl.append(image_information)
        elif lang == "en":
            images_en.append(image_information)

        img.close()
        if _iter > 3:
            break

    return images_pl, images_en


def check_lang(word_array: list[str], accuracy=0.1) -> str:
    """
    :param word_array: Lista słów
    :param accuracy: Dokładność jaka jest wystarczająca aby powiedzieć, że plik jest polski
    :return:
    """
    pl, en = 0, 0
    for word in word_array:
        try:
            det = Detector(word, quiet=True).languages[0].code
        except Exception:
            en += 1
            continue

        if det == 'pl':
            pl += 1
        elif det == 'en':
            en += 1

    language_parameter = 0 if en == 0 else pl / en
    return "pl" if language_parameter >= accuracy else "en"


def load_to_json():
    DIRPATH = "/home/rskay/PycharmProjects/pythonProject/Projekty/Fuckaton/datasets/train_set"
    for DIR in os.listdir(DIRPATH):
        if DIR == "resume" or DIR == "budget":
            continue
        print(DIR)
        imgpl, imgen = check_lang_dir(os.path.join(DIRPATH, DIR), verbose=True, label=True)

        with open(f"/home/rskay/PycharmProjects/pythonProject/Projekty/Fuckaton/images/{DIR}.json", "w") as f:
            json.dump({"pl": imgpl, "en": imgen}, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    load_to_json()
    # [('0059d81f-65ca-43f4-9842-4c1ad34e7481.tiff', 5), ('0099ed1d-42a6-4dc7-ac97-fe3bb86e3370.tiff', 1), ('00de9dce-de99-44e2-8165-f7a9bc523b1f.tiff', 15), ('01fc64bd-f99e-4419-964a-7340e8b7bd27.tiff', 15), ('02321721-6564-4814-b1cc-403117fc192e.tiff', 15), ('0236e4aa-1431-42f3-b938-b6825751c902.tiff', 15), ('029a4957-8a4c-476f-8feb-c3f0e444a4c6.tiff', 15), ('02e676e9-a01e-437c-81ba-9c7280fdcde9.tiff', 15), ('02f9e1f3-69c6-4f9b-84f8-c250c3c0c33e.tiff', 15), ('030075e9-ed70-4bf5-9e5b-cef2bf9923cf.tiff', 6), ('03888526-2497-4f03-99b9-3366a835b427.tiff', 1), ('03c34452-89cf-4e3d-802d-158b95404934.tiff', 15)]
