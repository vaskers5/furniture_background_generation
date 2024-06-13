import torch
import torch.nn.functional as F
from loguru import logger
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image


class ClipClassfifier:
    def __init__(self, device, data: dict[str, list[str]]):
        self.furniture_types = data["furniture_types"]
        self.model, self.preprocess = create_model_from_pretrained("hf-hub:timm/ViT-SO400M-14-SigLIP")
        self.tokenizer = get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP")
        self.cat_types = ["indoor", "outdoor"]
        self.furniture_categories = data["furniture_categories"]

    def describe_image(self, img):
        image = self.preprocess(img).unsqueeze(0)

        descriptions = [f"photo of {item}" for item in self.furniture_types]
        furnitrure_index = self.__get_clip_prediction(image, descriptions)

        furniture_description = descriptions[furnitrure_index]
        furniture_guess = self.furniture_types[furnitrure_index]
        
        category_list = [f"{cat} {furniture_description}" for cat in self.cat_types]
        cat_index = self.__get_clip_prediction(image, category_list)
        category_guess = self.cat_types[cat_index]

        furniture_cat_prompts = [f"photo of {item}" for item in self.furniture_categories]
        furniture_cat_index = self.__get_clip_prediction(image, furniture_cat_prompts)
        furniture_cat_guess = self.furniture_categories[furniture_cat_index]
        
        logger.info(f"Provided item classified as {furniture_guess} which must placed {category_guess}")
        return {"furniture": furniture_guess, "category": category_guess, "furniture_category": furniture_cat_guess}

    def __get_clip_prediction(self, image, item_list):
        text = self.tokenizer(item_list, context_length=self.model.context_length)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            text_probs = torch.sigmoid(
                image_features @ text_features.T * self.model.logit_scale.exp() + self.model.logit_bias
            )

        return text_probs[0].argmax().item()
