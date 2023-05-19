"""
Program dla podanego folderu ze zdjęciami zwraca listy pogrupowane po językach polskim i angielskim
Zwrócone wartości są w formatach list[dict], list[dict], gdzie dict zawiera informacje o:
    - Nazwie pliku : name
    - Listy z odczytanym tekstem ze zdjęcia : text
    Dla label = True lub type(int) także zwraca kategorię
"""

import os
import json

import pytesseract
from PIL import Image
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger

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
        except Exception:  # noqa
            en += 1
            continue

        if det == 'pl':
            pl += 1
        elif det == 'en':
            en += 1

    language_parameter = 0 if en == 0 else pl / en
    return "pl" if language_parameter >= accuracy else "en"


def load_to_json():
    dirpath = "/home/rskay/PycharmProjects/pythonProject/Projekty/Fuckaton/datasets/train_set"
    for directory in os.listdir(dirpath):
        if directory == "resume" or directory == "budget":
            continue
        print(directory)
        imgpl, imgen = check_lang_dir(os.path.join(dirpath, directory), verbose=True, label=True)

        with open(f"/home/rskay/PycharmProjects/pythonProject/Projekty/Fuckaton/images/{directory}.json", "w") as f:
            json.dump({"pl": imgpl, "en": imgen}, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    load_to_json()
