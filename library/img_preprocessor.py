from PIL import Image, ImageOps
import requests
from io import BytesIO
from transparent_background import Remover


class ImagePreprocessor:
    def __init__(self, device):
        self.remover = Remover(mode='fast', device=device, ckpt='weights/ckpt_fast.pth')
        self.resolution = (512, 512)
        
    def load_and_preprocess_img(self, orig_img: Image):
        resized_img = self.__resize_with_padding(orig_img, self.resolution)
        fg_mask = self.remover.process(resized_img, type='map')
        mask = ImageOps.invert(fg_mask)
        return {"orig_img": orig_img, "resized_img": resized_img, "mask": mask}

    @staticmethod
    def __resize_with_padding(img, expected_size):
        img.thumbnail((expected_size[0], expected_size[1]))
        delta_width = expected_size[0] - img.size[0]
        delta_height = expected_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        return ImageOps.expand(img, padding)
