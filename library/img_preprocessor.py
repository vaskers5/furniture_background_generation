from io import BytesIO

import requests
from PIL import Image, ImageOps
from transparent_background import Remover


class ImagePreprocessor:
    def __init__(self, device, data):
        self.remover = Remover(mode='fast', device=device, ckpt='weights/ckpt_fast.pth')
        self.resolution = (512, 512)
        self.data = data
        
    def load_and_preprocess_img(self, orig_img: Image, furniture_type):
        resized_img = self.__resize_with_padding(orig_img, self.resolution, self.data, furniture_type)
        fg_mask = self.remover.process(resized_img, type='map')
        mask = ImageOps.invert(fg_mask)
        return {"resized_img": resized_img, "mask": mask}

    @staticmethod
    def __resize_with_padding(img, expected_size, data: dict[str: list], furniture_type: str):

        resize_factor = 0.9375          # = object height / image height
        position_pt = 0.03125           # = top of the object position on image
        if furniture_type in data.keys():
            resize_factor, position_pt = list(map(float, data[furniture_type]))

        rectification_factor = 1 - resize_factor
        position_pt *= rectification_factor

        xy_factor = img.size[1] / img.size[0]
        y_size = resize_factor * expected_size[1]
        x_size = xy_factor * y_size
        x_size, y_size = int(x_size), int(y_size)
        print(f"resize orig immg to {(y_size, x_size)}")

        img = img.resize((y_size, x_size))

        top_pad = int(position_pt * expected_size[0])
        bottom_pad = expected_size[0] - top_pad - y_size
        l_pad = (expected_size[1] - x_size) // 2
        r_pad = expected_size[1] - l_pad - x_size
        padding = (l_pad, top_pad, r_pad, bottom_pad)

        return ImageOps.expand(img, padding).convert("RGB")
