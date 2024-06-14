import pyiqa
import torch
from PIL import Image


class IQARanker:
    def __init__(self, device: torch.device):
        self.metric_name = "clipiqa+_vitL14_512" # "maniqa-kadid"
        self.metric = pyiqa.create_metric(self.metric_name, device=device)

    def __call__(self, imgs: list[Image], num_infer_images: int):
        img_scores_data = [{"img": img, "metric": self.metric(img).item()} for img in imgs]
        img_scores_data = list(
            sorted(img_scores_data, key=lambda item: item["metric"], reverse=True)
        )  # bigger score -> better img
        result = [item["img"] for item in img_scores_data[:num_infer_images]]
        return result
