import json


from library.insert_everything import InsertEvetything


with open('data.json', 'r') as f:
    data = json.load(f)

insert_everything = InsertEvetything(data)
img_path = "test_imgs/handled_chair.webp"
insert_everything(img_path)
