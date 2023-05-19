import typing
import tensorflow as tf

from pathlib import Path


# tf.get_logger().setLevel('ERROR')


class Mod:
    """
    Klasa odpowiedzialna za trenowanie oraz zapisywanie sieci neuronowych
    """

    def __init__(self, dataset: typing.Union[str, Path],
                 input_shape: typing.Tuple[int, int] = (256, 256), *,
                 excluded: typing.Union[None, typing.List] = None) -> None:
        """
        Metoda odpowiedzialna za inicjalizację klasy 'Mod'

        :param dataset: Ścieżka do zbioru danych.
        :param input_shape: Rozmiar wejścia sieci neuronowej CNN.
        :param excluded: Ścieżki (podfoldery) wykluczone z trenowania sieci neuronowych.
        """
        self.excluded_dirs: typing.Union[None, typing.List] = excluded
        self.input_shape: typing.Tuple = input_shape
        image_dataset: 'Path' = Path(f'{dataset.__str__()}_jpg')
        text_dataset: 'Path' = Path(f'{dataset.__str__()}_txt')

        image_train_data, image_validation_data = self.load_image_data(image_dataset)
        text_train_data, text_validation_data = self.load_text_data(text_dataset)

        # self.save_cnn('v01')
        # self.model = self.concatenate_models(self.cnn_model, self.rnn_model)

        # self.fit_concatenated_model(image_train_data, text_train_data, image_validation_data, text_validation_data)

        self.rnn_model = self.build_rnn_model(text_train_data)
        self.save_rnn('v02')

        self.cnn_model = self.build_cnn_model()
        self.fit_cnn(image_train_data, image_validation_data)
        image_train_data, image_validation_data = self.load_image_data(image_dataset)
        self.save_cnn('v02')

    def save_cnn(self, output: typing.Union[str, 'Path']) -> None:
        """
        Metoda odpowiedzialna za zapis do pliku CNN

        :param output: Ścieżka do której ma zostać zapisany CNN
        :return: None
        """
        output = output if isinstance(output, Path) else Path(output)
        self.cnn_model.save(f'{output}_cnn.h5')

    def save_rnn(self, output: typing.Union[str, 'Path']) -> None:
        """
        Metoda odpowiedzialna za zapis do pliku RNN

        :param output: Ścieżka do której ma zostać zapisany RNN
        :return: None
        """
        output = output if isinstance(output, Path) else Path(output)
        self.rnn_model.save(f'{output}_rnn.tf', save_format='tf')

    @staticmethod
    def load_text_data(data_dir: 'Path') -> typing.Tuple['tf.data.Dataset', 'tf.data.Dataset']:
        """
        Metoda odpowiedzialna za wczytanie danych tekstowych ze zbioru danych.

        :param data_dir: Ścieżka do zbioru danych
        :return: Zbiory testowe oraz walidacyjne.
        """
        # classes = os.listdir(data_dir)
        return tf.keras.utils.text_dataset_from_directory(
            data_dir,
            batch_size=32,
            validation_split=0.2,
            subset='training',
            seed=2137), tf.keras.utils.text_dataset_from_directory(
            data_dir,
            batch_size=32,
            validation_split=0.2,
            subset='training',
            seed=2137)

    def load_image_data(self, data_dir: 'Path') -> typing.Tuple['tf.data.Dataset', 'tf.data.Dataset']:
        """
        Metoda odpowiedzialna za wczytanie ze zbioru danych oraz przetworzenie zdjęć na wektor cech.

        :param data_dir: Ścieżka do zbioru danych
        :return: Zbiory testowe oraz walidacyjne.
        """
        return tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=2137,
            image_size=self.input_shape,
            batch_size=32), tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=2137,
            image_size=self.input_shape,
            batch_size=32)

    @staticmethod
    def build_text_encoder(dataset: 'tf.data.Dataset') -> 'tf.keras.layers.TextVectorization':
        """
        Metoda odpowiedzialna za budowę enkodera tekstu
        :param dataset: zbiór danych na podstawie którego zostanie zbudowany enkoder

        :return: Warstwa odpowiedzialna za wektoryzację tekstu
        """
        encoder = tf.keras.layers.TextVectorization(max_tokens=4000)
        encoder.adapt(dataset.map(lambda txt, labels: txt))
        return encoder

    @staticmethod
    def concatenate_models(model1: 'tf.keras.Sequential', model2: 'tf.keras.Sequential') -> 'tf.keras.Sequential':
        """Metoda odpowiedzialna za konkatenację wartsw sieci neuronowych"""
        # model_concat = tf.keras.layers.concatenate([model1.output, model2.output], axis=-1)
        # model_concat = tf.keras.layers.Dense(16, activation='relu')(model_concat)
        # model = tf.keras.models.Model(inputs=[model1.input, model2.input], outputs=model_concat)

        # """models_concat = tf.keras.layers.Concatenate()[model1.output, model2.output]
        # models_concat = tf.keras.layers.Flatten()(models_concat)
        # merged = tf.keras.layers.Dense(2, activation="relu")(models_concat)
        # merged = tf.keras.layers.Dense(16)(merged)
        # model = tf.keras.models.Model(inputs=[model1.inputs, model2.inputs], outputs=merged)"""
        # model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #              metrics=['accuracy'])
        # return model
        raise NotImplementedError

    # def fit_concatenated_model(self, train_image_data: 'tf.data.Dataset', train_text_data: 'tf.data.Dataset',
    #                           validation_image_data: 'tf.data.Dataset',
    #                           validation_text_data: 'tf.data.Dataset') -> 'tf.keras.callbacks.History':
    #    print(type(train_image_data), type(train_text_data))
    #    return self.model.fit([train_image_data, train_text_data], epochs=10,
    #                          validation_data=[validation_image_data, validation_text_data])

    def build_rnn_model(self, dataset: 'tf.data.Dataset') -> 'tf.keras.Sequential':
        """
        Metoda odpowiedzialna za budowę sieci RNN

        :param dataset: Zbiór danych na podstawie którego ma zostać wytrenowana sieć
        :return: Model sieci neuronowej
        """
        encoder = self.build_text_encoder(dataset)
        rnn = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()),
                output_dim=64,
                mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(16)
        ])
        rnn.build(input_shape=(None,))
        rnn.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        return rnn

    def fit_rnn(self, train_data: 'tf.data.Dataset',
                validation_data: 'tf.data.Dataset') -> 'tf.keras.callbacks.History':
        """
        Metoda odpowiedzialna za dopasowanie sieci RNN do danych

        :param train_data: Dane (tekstowe) testowe
        :param validation_data: Dane (tekstowe) walidacyjne
        :return: History callback
        """
        return self.rnn_model.fit(train_data, epochs=10,
                                  validation_data=validation_data)

    def build_cnn_model(self) -> 'tf.keras.models.Sequential':
        """
        Metoda odpowiedzialna za budowę sieci CNN

        :param dataset: Zbiór danych na podstawie którego ma zostać wytrenowana sieć
        :return: Model sieci neuronowej
        """
        cnn = tf.keras.models.Sequential()
        cnn.add(tf.keras.layers.Rescaling(1. / 255))
        cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                       input_shape=self.input_shape))
        cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
        cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        cnn.add(tf.keras.layers.Flatten())
        cnn.add(tf.keras.layers.Dense(64, activation='relu'))
        cnn.add(tf.keras.layers.Dense(16))
        cnn.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        # cnn.build(input_shape=self.input_shape)
        return cnn

    def fit_cnn(self, train_data: 'tf.data.Dataset',
                validation_data: 'tf.data.Dataset') -> 'tf.keras.callbacks.History':
        """
        Metoda odpowiedzialna za dopasowanie sieci CNN do danych

        :param train_data: Dane (obrazy) testowe
        :param validation_data: Dane (obrazy) walidacyjne
        :return: History callback
        """
        return self.cnn_model.fit(train_data, epochs=10,
                                  validation_data=validation_data)


def main():
    excluded_dirs = ["pit37_v1", "umowa_sprzedazy_samochodu",
                     "umowa_o_dzielo",
                     "umowa_na_odleglosc_odstapienie",
                     "pozwolenie_uzytkowanie_obiektu_budowlanego"]
    Mod('../datasets/train_set', excluded=excluded_dirs)


if __name__ == '__main__':
    main()
