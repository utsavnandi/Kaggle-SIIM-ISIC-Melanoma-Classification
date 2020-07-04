import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

cfg = OmegaConf.load('./gpu_train_config.yaml')

DATA_DIR = cfg.data_dir
IMG_SIZE = cfg.FLAGS.img_size


class MelanomaDataset(Dataset):
    def __init__(
        self, df, labels, istrain=False, use_ce=False, transforms=None
    ):
        super().__init__()
        self.image_id = df["image_id"].values
        self.transforms = transforms
        self.labels = labels.values
        if not use_ce:
            self.neg_indices = np.where(self.labels == 0)[0]
            self.pos_indices = np.where(self.labels == 1)[0]
        else:
            self.neg_indices = np.where(self.labels[:, 0] == 1)[0]
            self.pos_indices = np.where(self.labels[:, 1] == 1)[0]
        self.istrain = istrain

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image, target = self.load_image(index)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image, target

    def load_image(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image_name = (
            DATA_DIR
            + f"512x512-dataset-melanoma/512x512-dataset-melanoma/{self.image_id[index]}.jpg"
        )
        image = cv2.imread(image_name, cv2.IMREAD_COLOR).astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        target = self.labels[index].astype(np.float32)
        return image, target


class MelanomaEvalDataset(Dataset):
    def __init__(self, df, labels, isEval=True, transform=None):
        super().__init__()
        self.image_id = df["image_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_name = (
            DATA_DIR + f"512x512-test/512x512-test/{self.image_id[index]}.jpg"
        )
        image = cv2.imread(image_name, cv2.IMREAD_COLOR).astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image
