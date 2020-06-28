import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = "/content/data/"
IMG_SIZE = 300


class MelanomaDataset(Dataset):
    def __init__(self, df, labels, istrain=False, use_ce=False, transforms=None):
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

        # if np.random.random() < 0.33:
        #    image, target = self.cutmix(image, target)

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
        target = self.labels[index].astype(np.float32)
        return image, target

    def cutmix(self, data, target, alpha=3):
        rand_index = self.get_rand_index()
        random_image, random_target = self.load_image(rand_index)
        h, w, _ = random_image.shape
        ncx, ncy = w / 2, h / 2
        img_shape = (IMG_SIZE, IMG_SIZE, 3)
        lam = np.random.beta(alpha, alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.shape, lam)
        data[bby1:bby2, bbx1:bbx2:] = random_image[bby1:bby2, bbx1:bbx2, :]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.shape[0] * data.shape[1]))
        new_target = lam * target + (1 - lam) * random_target
        return data, new_target

    def mixup(self, data, target, alpha=5):
        rand_index = self.get_rand_index()
        random_image, random_target = self.load_image(rand_index)
        lam = np.random.beta(alpha, alpha)
        data = data * lam + random_image * (1 - lam)
        data = data.astype(np.uint8)
        new_target = lam * target + (1 - lam) * random_target
        return data, new_target

    def get_rand_index(self):
        if np.random.random() > 0.5:
            rand_index = np.random.choice(self.pos_indices)
        else:
            rand_index = np.random.choice(self.neg_indices)
        return rand_index

    def rand_bbox(self, size, lam):
        W = size[1]
        H = size[0]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
