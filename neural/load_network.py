import typing

import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from vector import check_lang_dir, closest_class, load_norms


class Classifier:
    def __init__(self, data_path: typing.Union[str, Path], rnn_path: typing.Union[str, Path],
                 cnn_path: typing.Union[None, str, Path] = None, *,
                 norm_path: typing.Union[str, Path] = '../vector/normalized',
                 filename: str) -> None:
        """
        Metoda odpowiedzialna za inicjalizację instancji klasy 'Classifier'.

        :param data_path: Ścieżka do folderu ze zdjęciami do klasyfikacji
        :param rnn_path: Ścieżka do sieci RNN
        :param cnn_path: Ścieżka do sieci CNN
        :param norm_path: Ścieżka do unormowanych wartości wekterów odległości
        :param filename: Nazwa pliku do któego zostaną zapisane predykcje
        """
        self.rnn: 'tf.keras.models.Sequential' = tf.keras.models.load_model(rnn_path)
        self.cnn: typing.Optional['tf.keras.models.Sequential'] = tf.keras.models.load_model(
            cnn_path) if cnn_path else None
        self.data_path: 'Path' = Path(data_path) if isinstance(data_path, str) else data_path
        self.norm_path = norm_path
        self.filename: str = filename

        # self.classify_data()

    @staticmethod
    def filter_lang_data(path: typing.Union[str, Path]) -> \
            typing.Tuple[typing.List[typing.Dict], typing.List[typing.Dict]]:
        """
        Metoda odpowiedzialna za filtrowanie języka tekstu.

        :param path: Parametr określający ścieżkę pliku na podstawie którego będzie wykonywana klasyfikacja języka
        :return: typing.Tuple[typing.List[typing.Dict], typing.List[typing.Dict]]
        """
        return check_lang_dir(path, verbose=True)

    def classify_pl_data(self, data: typing.List[typing.Dict]) -> typing.List[typing.Tuple]:
        """
        Metoda odpowiedzialna za klasyfikację dokumentów oznaczonych jako dokumenty w języku polskim.

        :param data: Lista zawierająca dane tekstowe przeskanowanych dokumentów
        :return: Lista predykcji typów dokumentów.
        """
        norms = load_norms(self.norm_path)
        pred = []
        with open(self.filename, 'a+') as file:
            for doc in data:
                label = closest_class(doc['text'], norms)
                file.write(f'{doc["name"]},{label}\n')
                pred.append((doc['name'], label))
        return pred

    def classify_en_data(self, data: typing.List[typing.Dict]) -> typing.List[typing.Tuple]:
        """
        Metoda odpowiedzialna za klasyfikacją dokumentów oznaczonych jako dokumenty w języku angielskim.

        :param data: Lista zawierająca dane tekstowe przeskanowanych dokumentów
        :return: Lista predykcji typów dokumentów
        """
        pred = []
        to_cnn = []
        with open(self.filename, 'a+') as file:
            for doc in data:
                text = str(' '.join(doc['text']))
                if not text:
                    to_cnn.append(doc["name"])
                    continue
                # adapted_text = self.rnn.layers[0].adapt(text)
                class_name = self.rnn.predict(np.array([text]))
                label = np.argmax(class_name[0])
                label = label + 2 if label > 9 else label
                file.write(f'{doc["name"]},{label}\n')
                pred.append((doc['name'], label))

            for name in to_cnn:
                with Image.open(self.data_path / Path(name)) as img:
                    img = img.convert('RGB').resize((256, 256))
                    img = np.array(img)  # type: ignore
                    img = np.expand_dims(img, axis=0)
                    class_name = self.cnn.predict([img])
                    label = np.argmax(class_name[0])
                    label = label + 2 if label > 9 else label
                    file.write(f'{name},{label}\n')
                    pred.append((name, label))

        return pred

    def classify_data(self) -> typing.List[typing.Tuple]:
        """
        Metoda odpowiedzialna za klasyfikację dokumentów.

        :return: Lista sklasyfikowanych dokumentów
        """
        pred: typing.List[typing.Tuple] = []
        pl_data, en_data = self.filter_lang_data(self.data_path)
        # pl data classification
        pl_classified = self.classify_pl_data(pl_data)
        pred.extend(pl_classified)
        # en data classification
        en_classified = self.classify_en_data(en_data)
        pred.extend(en_classified)
        return pred


if __name__ == '__main__':
    # import threading

    classifier = Classifier('../datasets/test_set01', 'v02_rnn.tf', 'v01_cnn.h5', filename='results/results.txt')
    classifier.classify_data()
