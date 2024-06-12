import os
from pathlib import Path

import matplotlib.pyplot as plt
from RealESRGAN import RealESRGAN
from tqdm import tqdm
from PIL import Image


class PostProcessor:
    def __init__(self, device):
        self.upscaler = RealESRGAN(device, scale=4)
        self.upscaler.load_weights('weights/RealESRGAN_x4.pth', download=True)

    def __call__(self, images: list[Image], loc_folder: Path):
        
        result_images = []
        for idx, img in enumerate(tqdm(images, desc="Upscaling images")):
            sr_image = self.upscaler.predict(img)
            img_path = os.path.join(loc_folder, f"{idx + 1}.jpeg")
            sr_image.save(img_path, format="JPEG")
            result_images.append(sr_image)
            plot_path = os.path.join(loc_folder, "collage.png")
        
        # self.make_plot(result_images, plot_path)

    @staticmethod
    def make_plot(img_list, fig_path):
        fig = plt.figure(figsize=(20, 20))
        assert len(img_list) % 10 == 0
        column_num = 5 
        raws_num = len(img_list) // column_num
        for i, img in enumerate(img_list):
            img = img.resize((512, 512))
            ax = fig.add_subplot(4, 5, i + 1)
            ax.set_title(f"img_number_{i + 1}")
            ax.imshow(img)
            ax.axis('off')
            
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.savefig(fig_path)
        
        plt.show()
