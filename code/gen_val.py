import os
import json
import random
import os.path as pth
import shutil

with open('train/train.json') as f:
    j = json.load(f)

images = j['images']
annos = j['annotations']
cats = j['categories']

# 2517 train => 2217 train & 300 val
val_images = random.sample(images, 300)
val_id = [i['id'] for i in val_images]

for image in val_images:
    images.remove(image)
train_images = images
train_id = [i['id'] for i in train_images]

val_annos = [ann for ann in annos if ann['image_id'] in val_id]
train_annos = [ann for ann in annos if ann['image_id'] in train_id]

val_json = {'images': val_images, 'annotations': val_annos, 'categories': cats}
train_json = {'images': train_images, 'annotations': train_annos, 'categories': cats}

os.mkdir('val')
with open('val.json', 'w') as f:
    json.dump(val_json, f)

with open('train.json', 'w') as f:
    json.dump(train_json, f)

os.mkdir('val/val')
for i in val_images:
    fname = i['file_name']
    pth_orig = pth.join('train/train', fname)
    pth_fin = pth.join('val/val', fname)
    shutil.move(pth_orig, pth_fin)
    




