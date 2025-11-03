import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import albumentations as A
import random



# 定义增强操作
class RandomScaleCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        scale = random.uniform(0.8, 1.2)  # 随机缩放因子
        width, height = img.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height))

        # 确保缩放后的尺寸至少等于目标尺寸
        new_width = max(new_width, self.size[0])
        new_height = max(new_height, self.size[1])

        img = img.resize((new_width, new_height))

        # 随机裁剪
        left = random.randint(0, new_width - self.size[0])
        top = random.randint(0, new_height - self.size[1])
        img = img.crop((left, top, left + self.size[0], top + self.size[1]))
        return img


class RandomScaleRotate:
    def __init__(self, fillcolor=0):
        self.fillcolor = fillcolor

    def __call__(self, img):
        scale = random.uniform(0.8, 1.2)  # 随机缩放因子
        width, height = img.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height))

        angle = random.uniform(-30, 30)  # 随机旋转角度
        img = img.rotate(angle, fillcolor=self.fillcolor)
        return img


class ElasticTransform:
    def __init__(self, alpha=1, sigma=1):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        img_np = np.array(img)
        transform = A.ElasticTransform(alpha=self.alpha, sigma=self.sigma)
        img_np = transform(image=img_np)["image"]
        return Image.fromarray(img_np)


# 组合所有的弱增强操作
class WeakAugment:
    def __init__(self, img_size, fillcolor=0):
        self.transforms = transforms.Compose([
            RandomScaleCrop(img_size),
            RandomScaleRotate(fillcolor=fillcolor),
            transforms.RandomHorizontalFlip(p=0.5),
            ElasticTransform()
        ])

    def __call__(self, img):
        return self.transforms(img)


# 读取图片并应用弱增强
def apply_weak_augmentation(image_path, img_size=(384, 384), fillcolor=255):
    img = Image.open(image_path)
    weak_aug = WeakAugment(img_size, fillcolor)
    img_aug = weak_aug(img)
    return img_aug


class RandomBrightness:
    def __init__(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v

    def __call__(self, img):
        brightness_factor = random.uniform(self.min_v, self.max_v)
        return transforms.functional.adjust_brightness(img, brightness_factor)


class RandomContrast:
    def __init__(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v

    def __call__(self, img):
        contrast_factor = random.uniform(self.min_v, self.max_v)
        return transforms.functional.adjust_contrast(img, contrast_factor)


class GaussianBlur:
    def __init__(self, kernel_size, num_channels):
        self.kernel_size = kernel_size
        self.num_channels = num_channels

    def __call__(self, img):
        # Applying Gaussian blur
        return transforms.GaussianBlur(kernel_size=self.kernel_size)(img)


def get_strong_augmentation(min_v, max_v, img_size, num_channels):
    return transforms.Compose([
        RandomBrightness(min_v, max_v),
        RandomContrast(min_v, max_v),
        GaussianBlur(kernel_size=int(0.1 * img_size), num_channels=num_channels),
    ])


# 示例用法
if __name__ == "__main__":
    img_path = r"D:\Project\SynFoC-master\code\generate_images\00_00.png"  # 替换为图片的路径
    augmented_img = apply_weak_augmentation(img_path)
    augmented_img.save("augmented_weak_image.jpg")  # 保存增强后的图片
