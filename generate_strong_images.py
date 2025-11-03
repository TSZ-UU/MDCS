from PIL import Image
import random
from torchvision import transforms


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
        # Ensure kernel_size is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1  # Make it odd if it is even
        return transforms.GaussianBlur(kernel_size=self.kernel_size)(img)


def get_strong_augmentation(min_v, max_v, img_size, num_channels):
    return transforms.Compose([
        RandomBrightness(min_v, max_v),
        RandomContrast(min_v, max_v),
        GaussianBlur(kernel_size=int(0.1 * img_size), num_channels=num_channels),
    ])


# 加载图像
img_path = r'D:\Project\SynFoC-master\code\data\ProstateSlice\BMC\test\image\00_06.png'  # 请替换为图像文件路径
img = Image.open(img_path)

min_v = 0.7
max_v = 1.3
img_size = 224  # 根据实际图像尺寸设置
num_channels = 3  # 通常为3，如果是灰度图则为1

strong_augmentation = get_strong_augmentation(min_v, max_v, img_size, num_channels)

# 使用强增强
augmented_img = strong_augmentation(img)

# # 显示结果（可选）
# augmented_img.show()
augmented_img.save("augmented_strong_image.jpg")  # 保存增强后的图片
