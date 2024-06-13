import os
import shutil

import torch
from loguru import logger
from tqdm import tqdm
from PIL import Image

from library.clip_classifier import ClipClassfifier
from library.img_preprocessor import ImagePreprocessor
from library.post_processor import PostProcessor
from library.sd_worker import SdWorker



class InsertEvetything:
    def __init__(self, data: dict[str, list[str]]):
        self.data = data
        device = torch.device("cuda:0")
        self.preprocessor = ImagePreprocessor(device, self.data["heuristics"])
        self.clip_clf = ClipClassfifier(device, self.data["furniture_types"])
        self.post_proc = PostProcessor(device)
        self.sd_worker = SdWorker(device)

    def __call__(self, img: Image) -> None:
        
        raw_img = img.copy()
        item_description = self.clip_clf.describe_image(raw_img)
        preproc_data = self.preprocessor.load_and_preprocess_img(img, item_description['furniture'])
        
        all_generated_images = []
        
        available_locations = self.data[item_description["category"]]
        
        for loc_idx, location in enumerate(tqdm(available_locations, desc="Processing locations")):
            
            prompt = f"{item_description['furniture']} {location}"
            logger.info(f"Current generation prompt is: '{prompt}'")
            loc_folder = os.path.join("results", str(loc_idx))
            
            shutil.rmtree(loc_folder, ignore_errors=True)
            os.makedirs(loc_folder, exist_ok=True)
            
            pipe_images = self.sd_worker(prompt, preproc_data)
            all_generated_images.extend(pipe_images)

        return all_generated_images
