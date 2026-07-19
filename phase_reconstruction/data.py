import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

class PhaseDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        self.imgs_path1 = sorted(glob.glob(os.path.join(data_path, 'frame1/*.png')))
        self.imgs_path2 = sorted(glob.glob(os.path.join(data_path, 'frame2/*.png')))
        self.labels = sorted(glob.glob(os.path.join(data_path, 'phi/*.png')))
        counts = (len(self.imgs_path1), len(self.imgs_path2), len(self.labels))
        if not all(counts):
            raise ValueError(f"Expected PNG files in frame1/, frame2/, and phi/; got {counts}.")
        if len(set(counts)) != 1:
            raise ValueError(f"Dataset folders have different file counts: {counts}.")

    def add_dead_pixels(self, img, rng, ratio=0.001):
        h, w = img.shape[1], img.shape[2]
        num = int(h * w * ratio)
        ys = rng.randint(0, h, num)
        xs = rng.randint(0, w, num)
        vals = rng.randint(0, 256, num)
        img = img.copy()
        img[0, ys, xs] = vals
        return img

    def add_random_block(self, img, rng, max_size=20):
        h, w = img.shape[1], img.shape[2]
        block_size = rng.randint(5, max_size)
        y = rng.randint(0, h - block_size)
        x = rng.randint(0, w - block_size)

        mode = rng.choice(['black', 'white', 'noise'])
        img = img.copy()
        if mode == 'black':
            img[0, y:y+block_size, x:x+block_size] = 0
        elif mode == 'white':
            img[0, y:y+block_size, x:x+block_size] = 255
        else:
            img[0, y:y+block_size, x:x+block_size] = rng.randint(
                0, 256, (block_size, block_size)
            )
        return img

    def __getitem__(self, index):
        # DataLoader seeds NumPy independently in every worker. Using that
        # evolving random stream keeps a run reproducible without freezing a
        # sample to exactly the same augmentation on every epoch.
        rng = np.random

        img1 = cv2.imread(self.imgs_path1[index], -1)
        img2 = cv2.imread(self.imgs_path2[index], -1)
        label = cv2.imread(self.labels[index], -1)

        if img1 is None or img2 is None or label is None:
            raise OSError(f"Failed to read dataset item at index {index}.")

        if img1.ndim == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if img2.ndim == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if label.ndim == 3: label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        img1 = img1[None]
        img2 = img2[None]
        label = label[None]

        # Preserve clean inputs before applying augmentation.
        img1_clean = img1.copy()
        img2_clean = img2.copy()

        # Apply reproducible stream-based augmentation. The result changes on
        # later visits to this sample instead of being fixed by its index.
        if rng.random() < 0.5:
            img1 = self.add_dead_pixels(img1, rng)
        if rng.random() < 0.5:
            img2 = self.add_dead_pixels(img2, rng)

        if rng.random() < 0.3:
            img1 = self.add_random_block(img1, rng, max_size=30)
        if rng.random() < 0.3:
            img2 = self.add_random_block(img2, rng, max_size=30)

        divisor = 65535.0 if img1.max() > 255 else 255.0

        img1 = img1 / divisor
        img2 = img2 / divisor
        img1_clean = img1_clean / divisor
        img2_clean = img2_clean / divisor
        label = label / divisor

        image_noisy = np.concatenate([img1, img2], axis=0)
        image_clean = np.concatenate([img1_clean, img2_clean], axis=0)

        return (
            image_noisy.astype(np.float32),
            image_clean.astype(np.float32),
            label.astype(np.float32)
        )

    def __len__(self):
        return len(self.imgs_path1)
