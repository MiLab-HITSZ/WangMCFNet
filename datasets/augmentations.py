from albumentations import *
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import random

class CutoutOld(ImageOnlyTransform):
    def __init__(self, num_holes=1, max_h_size=32, max_w_size=32, fill_value=0, always_apply=False, p=0.5):
        # super(CutoutOld, self).__init__(always_apply, p)
        super(CutoutOld, self).__init__(always_apply=always_apply, p=p)

        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def apply(self, image, **params):
        h, w = image.shape[:2]
        for _ in range(self.num_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)
            y1 = np.clip(y - self.max_h_size // 2, 0, h)
            y2 = np.clip(y + self.max_h_size // 2, 0, h)
            x1 = np.clip(x - self.max_w_size // 2, 0, w)
            x2 = np.clip(x + self.max_w_size // 2, 0, w)
            image[y1:y2, x1:x2] = self.fill_value
        return image


class JpegCompressionOld(ImageOnlyTransform):

    def __init__(self, quality_lower=30, quality_upper=90, always_apply=False, p=0.5):
        super(JpegCompressionOld, self).__init__(always_apply=always_apply, p=p)
        # super(JpegCompressionOld, self).__init__(always_apply, p)
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def apply(self, img, **params):
        quality = np.random.randint(self.quality_lower, self.quality_upper)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if result:
            img = cv2.imdecode(encimg, 1)
        return img

from imgaug import augmenters as iaa
# IAAAdditiveGaussianNoise = iaa.AdditiveGaussianNoise

augment0=Compose([HorizontalFlip()],p=1)
from albumentations import (
    Compose, HorizontalFlip, RandomResizedCrop, CenterCrop,
    HueSaturationValue, RandomBrightnessContrast,
    GaussNoise, GaussianBlur, MotionBlur,
    # JpegCompression,
    # Cutout,
    OneOf, CLAHE, ToGray
)

augment1 = Compose([
    # 先做随机裁剪 +缩放（在这一步就统一尺寸）
    # RandomResizedCrop(size=(320, 320), scale=(0.8, 1.0), ratio=(0.9, 1.1)),

    # RandomResizedCrop(height=320, width=320, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    # RandomResizedCrop(height=320, width=320, scale=(0.8, 1.0), ratio=(0.9, 1.1), size=(320, 320)),
        # 基本几何变换
    HorizontalFlip(p=0.5),

    # 颜色 /光照变换
    HueSaturationValue(p=0.5, hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20),
    RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
    CLAHE(p=0.3),  # 局部增强

    # Noise / 模糊 /压缩增强
    OneOf([
        GaussNoise(var_limit=(5.0, 30.0)),
        GaussianBlur(blur_limit=(3, 7)),
        MotionBlur(blur_limit=5),
    ], p=0.3),

    JpegCompressionOld(quality_lower=30, quality_upper=90, p=0.5),

    # 遮挡 / cutout
    CutoutOld(num_holes=4, max_h_size=50, max_w_size=50, fill_value=0, p=0.4),

    # 转灰 /抽象处理（有时候让模型不完全依赖颜色）
    ToGray(p=0.1),

], additional_targets={'mask': 'mask'})

augmentations={'augment0':augment0,'augment1':augment1 }
# augmentations={'augment0':augment0}
