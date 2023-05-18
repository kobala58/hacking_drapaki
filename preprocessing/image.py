import typing
from pathlib import Path
from PIL import Image


def preprocess_image(path: typing.Union[Path, str], *, threshold: int = 170) -> 'Image':
    with Image.open(path) as img:
        img = img.convert('L')
        img_conv = img.point(lambda x: 255 if x > threshold else 0, mode='1')
        # img_conv = img_conv.resize((530, 700))
        # layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))
    return img_conv
