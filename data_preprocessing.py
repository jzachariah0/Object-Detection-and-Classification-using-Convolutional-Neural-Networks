import os
import cv2

# Set up directories
src_dir = '/path/to/source/dataset'
dst_dir = '/path/to/destination/dataset'
os.makedirs(dst_dir, exist_ok=True)

# Resize images and save them to new directory
for subdir in os.listdir(src_dir):
    if not os.path.isdir(os.path.join(src_dir, subdir)):
        continue
    os.makedirs(os.path.join(dst_dir, subdir), exist_ok=True)
    for file in os.listdir(os.path.join(src_dir, subdir)):
        img = cv2.imread(os.path.join(src_dir, subdir, file))
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(os.path.join(dst_dir, subdir, file), img)
