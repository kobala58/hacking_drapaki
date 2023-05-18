import os
import json
# import pickle
# import shutil
import typing

import numpy as np
import pytesseract
import tensorflow as tf
# import tensorflow_hub as hub
# import tensorflow_text as text
# import matplotlib.pyplot as plt
# from official.nlp import optimization  # to create AdamW optimizer

from pathlib import Path
from PIL import Image
from preprocessing import preprocess_image

tf.get_logger().setLevel('ERROR')


class Mod:
    def __init__(self, words_path: typing.Union[Path, str], input_shape: typing.Tuple[int, int] = (256, 256)):
        self.input_shape = input_shape
        if isinstance(words_path, str):
            words_path = Path(words_path)
        # for directory in os.listdir(words_path):
        self.words = {}
        for file in os.listdir(words_path):
            with open(words_path / file, 'r') as data:
                self.words[file.__str__().split('.')[0]] = json.load(data)

        # print(self.words)

    def prepare_image(self, img_path: typing.Union[Path, str]) -> np.ndarray:
        with Image.open(img_path) as img:
            return tf.image.resize(img, self.input_shape)

    def create_model(self) -> 'tf.keras.models.Sequential':
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Rescaling(1. / 255))
        model.add(tf.keras.layers.Conv2D(None, (3, 3), activation='relu'),
                  input_shape=self.input_shape)
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(16))
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy', 'f1'])
        return model

        # layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3))

    def fit_model(self):
        """
        history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
        """
        raise NotImplementedError

    @staticmethod
    def load_data(data_path: typing.Union[Path, str]) -> typing.List[Path]:
        if isinstance(data_path, str):
            data_path = Path(data_path)
        files = []
        for directory in os.listdir(data_path):
            directory = Path(directory)
            for file in os.listdir(data_path / directory):
                files.append(data_path / directory / Path(file))
        return files

    def split_data(self, data: typing.List[Path]) -> typing.Tuple[typing.List]:
        raise NotImplementedError

    @staticmethod
    def levenstein(w1: str, w2: str) -> float:
        raise NotImplementedError

    @staticmethod
    def ocr_data(data: typing.List[Path]) -> typing.Dict:
        """funkcja zwracajaca top slowa z ocr'a"""
        image_text = {}
        for file in data:
            img = preprocess_image(file)
            img.show()
            ret = pytesseract.image_to_string(img)
            image_text[file.__str__()] = ret
            data = [word for word in ret.strip().lower().split() if len(word) > 4]
            # todo: odleglosc levensteina
            words = {}
            for x in data:
                if x in words.keys():
                    words[x] += 1
                else:
                    words[x] = 1
        return image_text
        # raise NotImplementedError

    def __find_match(self, text_list: list[str]) -> str:
        for doc, words in self.words:
            pass
        return ''

    def find_match(self, data: dict) -> dict:
        matches = {}
        for key, val in data.items():
            match = self.__find_match(val)
            matches[key] = match
        return matches


def main():
    net = Mod('../words')
    # net.load_data('../datasets/train_set')
    print(net.ocr_data(
        [Path('../datasets/train_set/umowa_o_dzielo/2929e51f-aeb8-4af1-8dab-dc15e1ff8978.jpg')]))


if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    main()
