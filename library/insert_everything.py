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
from library.ollama_client import OllamaClient


class InsertEvetything:
    def __init__(self, data: dict[str, list[str]], prompt_generator="default"):
        self.data = data
        device = torch.device("cuda:0")
        self.preprocessor = ImagePreprocessor(device, self.data["heuristics"])
        self.clip_clf = ClipClassfifier(device, self.data["clip_data"])
        self.post_proc = PostProcessor(device)
        self.sd_worker = SdWorker(device)
        self.iqa_ranker = IQARanker(device)
        self.prompt_generator = prompt_generator
        self.llama_client = OllamaClient()

    def __generate_prompts(self, clip_data: dict[str, str], location_category: str) -> list[str]:
        prompts = []
        if self.prompt_generator == "default":
            
            if location_category in ["indoor", "outdoor"]:
                available_locations = self.data[location_category]
    
            elif location_category == "all":
                available_locations = [*self.data["indoor"], *self.data["outdoor"]]
    
            elif location_category == "automatic":
                available_locations = self.data[clip_data["category"]]
                
            prompts = [f"{clip_data['furniture']} {location}" for location in available_locations]
            
        elif self.prompt_generator == "llama":
            furniture_category_mapper = {
                                        "indoor": "house furniture",
                                        "outdoor": "garden furniture",
                                        "all": clip_data["furniture_category"],
                                        "automatic": clip_data["furniture_category"]
                                        }
            llama_prompt = f'{clip_data["furniture"]}, {clip_data["furniture_category"]}'
            llama_result = self.llama_client.generate(llama_prompt)
            prompts = [item[3:].replace('.', '') for item in llama_result.split('\n')]
    
        return prompts
            
        

    def __call__(self, img: Image, num_result_imgs: int, location_category: str="automatic") -> None:

        raw_img = img.copy()
        item_description = self.clip_clf.describe_image(raw_img)
        preproc_data = self.preprocessor.load_and_preprocess_img(img, item_description['furniture'])
        all_generated_images = []
        prompts = self.__generate_prompts(item_description, location_category)
        
        for loc_idx, prompt in enumerate(tqdm(prompts, desc="Processing locations")):
            logger.info(f"Current generation prompt is: '{prompt}'")
            pipe_images = self.sd_worker(prompt, preproc_data)
            all_generated_images.extend(pipe_images)

        pipe_images = self.iqa_ranker(all_generated_images, num_infer_images=num_result_imgs)
        
        result_images = self.post_proc(pipe_images)
        return result_images
