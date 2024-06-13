import os
import shutil
from typing import Callable

import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm
import asyncio

from library.clip_classifier import ClipClassfifier
from library.img_preprocessor import ImagePreprocessor
from library.iqa_ranker import IQARanker
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
        self.iqa_ranker = IQARanker(device)

    def __call__(self, img: Image, num_result_imgs: int, location_category: str, progress_callback: Callable) -> None:

        raw_img = img.copy()
        item_description = self.clip_clf.describe_image(raw_img)
        preproc_data = self.preprocessor.load_and_preprocess_img(img, item_description['furniture'])
        all_generated_images = []

        if location_category in ["indoor", "outdoor"]:

            available_locations = self.data[location_category]

        elif location_category == "all":
            available_locations = [*self.data["indoor"], *self.data["outdoor"]]

        elif location_category == "automatic":
            available_locations = self.data[item_description["category"]]

        total_operations = len(available_locations) + 2  # ranking + post-proc

        for loc_idx, location in enumerate(tqdm(available_locations, desc="Processing locations")):
            prompt = f"{item_description['furniture']} {location}"
            logger.info(f"Current generation prompt is: '{prompt}'")

            pipe_images = self.sd_worker(prompt, preproc_data)
            all_generated_images.extend(pipe_images)

        pipe_images = self.iqa_ranker(all_generated_images, num_infer_images=num_result_imgs)
        
        result_images = self.post_proc(pipe_images)
        return result_images
