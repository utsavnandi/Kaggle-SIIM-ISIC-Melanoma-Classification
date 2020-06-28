import numpy as np

#import cv2
import albumentations as A
from albumentations.augmentations import functional as FA
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch.transforms import ToTensorV2

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class GridMask(DualTransform):
    def __init__(
        self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5
    ):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones(
                    (int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))
                ).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                            int(i * grid_h) : int(i * grid_h + grid_h / 2),
                            int(j * grid_w) : int(j * grid_w + grid_w / 2),
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w),
                            ] = self.fill_value

                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = FA.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:, :, np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h : rand_h + h, rand_w : rand_w + w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = (
            np.random.randint(self.rotate[0], self.rotate[1])
            if self.rotate[1] > 0
            else 0
        )

        return {"mask": mask, "rand_h": rand_h, "rand_w": rand_w, "angle": angle}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("num_grid", "fill_value", "rotate", "mode")


def get_train_transforms(p=1.0, IMG_SIZE=300):
    return A.Compose(
        [
            A.OneOf(
                [
                    A.CenterCrop(3 * IMG_SIZE // 4, 3 * IMG_SIZE // 4, p=0.5),
                    A.CenterCrop(4 * IMG_SIZE // 5, 4 * IMG_SIZE // 5, p=0.5),
                ],
                p=0.5,
            ),
            A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, always_apply=True, p=1),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            GridMask(num_grid=(4, 9), rotate=30, p=0.5),
            A.OneOf(
                [A.MedianBlur(blur_limit=3, p=0.5), A.Blur(blur_limit=3, p=0.5),], p=0.5
            ),
            A.ShiftScaleRotate(
                interpolation=1,
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.33,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.33),
                    A.GridDistortion(p=0.33),
                    A.IAAPiecewiseAffine(p=0.33),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                p=0.5,
            ),
            A.CLAHE(clip_limit=(1, 3), p=0.33),
            A.MultiplicativeNoise(multiplier=[0.9, 1.1], elementwise=True, p=0.5),
            A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            ToTensorV2(p=1.0),
        ],
        p=p,
    )


def get_valid_transforms(IMG_SIZE=300):
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE, interpolation=2, always_apply=True, p=1),
            A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            ToTensorV2(p=1.0),
        ]
    )
