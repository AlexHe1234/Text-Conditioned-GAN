from torch.utils.data import Dataset
import os
import cv2
import torch.nn.functional as F
import torch


class Intel(Dataset):
    def __init__(self, dataset_dir, train=True):
        super().__init__()
        if not train:
            raise NotImplementedError
        self.classes = {'buildings': 0,
                 'forest': 1,
                 'glacier': 2,
                 'mountain': 3,
                 'sea': 4,
                 'street': 5}
        self.data = []
        self.dir = dataset_dir
        for folder in self.classes.keys():
            path = os.path.join(self.dir, 'seg_train/seg_train', folder)
            files = os.listdir(path)
            for file in files:
                self.data.append({'path': os.path.join(path, file), 'class': self.classes[folder]})

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def condition(img):
        img_gaussian = cv2.GaussianBlur(img, [3, 3], 1)
        img_canny = cv2.Canny(img_gaussian, 100, 200)
        return img_canny

    def __getitem__(self, index):
        data_i = self.data[index]
        img = cv2.imread(data_i['path'], cv2.IMREAD_COLOR)
        encode = F.one_hot(torch.tensor(data_i['class']), len(self.classes.keys()))
        img_input = self.condition(img)
        return {'input': img_input, 
                'condition': encode,
                'og': img}


def make_dataset(cfg):
    return Intel(cfg.dataset_dir, train=True)
