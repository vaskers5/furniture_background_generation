from diffusers import DiffusionPipeline
import torch
from PIL import Image
import numpy as np


class SdWorker:
    def __init__(self, device):
        self.model_id = "yahoo-inc/photo-background-generation"
        self.pipeline = DiffusionPipeline.from_pretrained(self.model_id, custom_pipeline=self.model_id)
        self.pipeline = self.pipeline.to(device)
        self.num_images_per_prompt=20
        self.seed = 42
        self.cond_scale = 1.0
        self.generator = torch.Generator(device='cuda').manual_seed(self.seed)
        self.num_steps = 25

    def __call__(self, prompt: str, preproc_data: dict[str, Image]) -> list[Image]:
        with torch.autocast('cuda'):
            pipe_images = self.pipeline(
                prompt=prompt, 
                image=preproc_data["resized_img"], mask_image=preproc_data["mask"],
                control_image=preproc_data["mask"], num_images_per_prompt=self.num_images_per_prompt,
                generator=self.generator, num_inference_steps=self.num_steps,
                guess_mode=False,
                controlnet_conditioning_scale=self.cond_scale
            ).images
        return pipe_images
