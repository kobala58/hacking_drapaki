import os
import typing
from pathlib import Path
from PIL import Image


def convert_to_jpg(data_path: typing.Union[str, Path]) -> None:
    """
    Funkcja odpowiedzialna za wstępne przetworzenie danych

    :param data_path: Ścieżka danych które mają zostać przetworzone
    :return: None
    """
    # dataset = '../datasets/train_set'
    dataset_jpg = f'{data_path}_jpg'
    if not os.path.exists(dataset_jpg):
        os.makedirs(Path(dataset_jpg))
    for directory in os.listdir(data_path):
        for filename in os.listdir(Path(data_path) / Path(directory)):
            with Image.open(Path(data_path) / Path(directory) / Path(filename), 'r') as img:
                img = img.resize((256, 256))
                if not os.path.exists(Path(dataset_jpg) / Path(directory)):
                    os.makedirs(Path(dataset_jpg) / Path(directory))
                img.save(Path(dataset_jpg) / Path(directory) / Path(filename.split('.')[0] + '.jpg'),
                         'JPEG')
