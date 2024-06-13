import json
from PIL import Image

from library.insert_everything import InsertEvetything


with open('data.json', 'r') as f:
    data = json.load(f)

if __name__ == "__main__":
    insert_everything = InsertEvetything(data)
    img_path = "test_imgs/table.webp"
    orig_img = Image.open(img_path)
    imgs = insert_everything(orig_img, 5, 'indoor', None)
    print(len(imgs))
    for idx, img in enumerate(imgs):
        img.save(f"img_{idx}.png")
