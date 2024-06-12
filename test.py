import json
from PIL import Image

from library.insert_everything import InsertEvetything


with open('data.json', 'r') as f:
    data = json.load(f)

if __name__ == "__main__":
    insert_everything = InsertEvetything(data)
    img_path = "test_imgs/handled_chair.webp"
    orig_img = Image.open(img_path)
    insert_everything(orig_img)
