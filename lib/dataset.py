import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def read_image(img):
    img = np.array(img)
    width = img.shape[1]
    width_half = width // 2
    input_image = img[:, :width_half, :]
    target_image = img[:, width_half:, :]
    input_image = input_image.astype(np.float32)
    target_image = target_image.astype(np.float32)
    return input_image, target_image


def random_crop(img, dim):
    height, width, _ = dim
    x = np.random.uniform(low=0, high=int(height - 256))
    y = np.random.uniform(low=0, high=int(height - 256))
    return img[:, int(x):int(x) + 256, int(y):int(y) + 256]


def random_jittering_mirroring(input_image, target_image, height=286, width=286):
    input_image = cv2.resize(input_image, (height, width), interpolation=cv2.INTER_NEAREST)
    target_image = cv2.resize(target_image, (height, width), interpolation=cv2.INTER_NEAREST)
    stacked_image = np.stack([input_image, target_image], axis=0)
    cropped_image = random_crop(stacked_image, dim=[height, width, 3])
    input_image, target_image = cropped_image[0], cropped_image[1]
    if np.random.rand() > 0.5:
        input_image = np.fliplr(input_image)
        target_image = np.fliplr(target_image)
    return input_image, target_image


def normalize(input_image, target_image):
    input_image = input_image / 127.5 - 1
    target_image = target_image / 127.5 - 1
    return input_image, target_image


class Train(object):
    def __init__(self, cfg):
        self.cfg = cfg
        
    def __call__(self, image):
        input_image, target_image = read_image(image)
        input_image, target_image = random_jittering_mirroring(input_image, target_image)
        input_image, target_image = normalize(input_image, target_image)
        image_a = torch.from_numpy(input_image.copy().transpose((2, 0, 1)))
        image_b = torch.from_numpy(target_image.copy().transpose((2, 0, 1)))
        return image_a, image_b


def make_dataloader(cfg):
    # read in the sets and preprocess them
    train_dataset = ImageFolder(cfg.dataset_dir, transform=Train(cfg))
    return  DataLoader(train_dataset, cfg.batch_size, shuffle=True)
