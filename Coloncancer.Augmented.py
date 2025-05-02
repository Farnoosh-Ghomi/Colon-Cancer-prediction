#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# 1) Input and output directories for each class
normal_input_dir = "/content/drive/MyDrive/Coloncancer/Normals"
cancer_input_dir = "/content/drive/MyDrive/Coloncancer/Cancers"

normal_output_dir = "/content/drive/MyDrive/Coloncancer_Augmented/Normals"
cancer_output_dir = "/content/drive/MyDrive/Coloncancer_Augmented/Cancers"

# 2) Desired final number of images per class
desired_count = 1500

# 3) Define data augmentation pipeline
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=50.0,
    fill_mode='reflect'
)

def augment_class(input_dir, output_dir, desired_count_per_class):
    os.makedirs(output_dir, exist_ok=True)
    # List input images
    imgs = [f for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    orig_count = len(imgs)
    aug_needed = desired_count_per_class - orig_count
    if aug_needed <= 0:
        print(f"[{os.path.basename(output_dir)}] No augmentation needed (count: {orig_count})")
        return

    # How many augmentations per image
    n_per_img = math.ceil(aug_needed / orig_count)
    print(f"[{os.path.basename(output_dir)}] Original: {orig_count} | Needed: {aug_needed} | Per image: {n_per_img} augmentations")

    total_generated = 0
    for img_name in imgs:
        if total_generated >= aug_needed:
            break
        img_path = os.path.join(input_dir, img_name)
        x = img_to_array(load_img(img_path))
        x = x.reshape((1,) + x.shape)

        prefix = os.path.splitext(img_name)[0]
        gen = datagen.flow(
            x,
            batch_size=1,
            save_to_dir=output_dir,
            save_prefix=prefix,
            save_format='png'
        )
        for _ in range(n_per_img):
            if total_generated >= aug_needed:
                break
            next(gen)
            total_generated += 1

    print(f"► [{os.path.basename(output_dir)}] Generated: {total_generated} new images. Total: {orig_count + total_generated}")

# 4) Run augmentation for each class
augment_class(normal_input_dir, normal_output_dir, desired_count)
augment_class(cancer_input_dir, cancer_output_dir, desired_count)

print(" Done! Two separate folders with augmented images are ready:")
print(f"  • Normals  ➔ {normal_output_dir}")
print(f"  • Cancers  ➔ {cancer_output_dir}")

