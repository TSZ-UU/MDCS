from PIL import Image, ImageOps
import numpy as np

# 加载图片
image1 = Image.open(r'E:\SynFoC-master\generate_images\RDR\pred_domain1_1750752669708.png').convert("RGBA")
image2 = Image.open(r'E:\SynFoC-master\generate_images\RDR\pred_domain1_1750752782303.png').convert("RGBA")

# 获取图片的尺寸，假设两张图片尺寸相同
width, height = image1.size

# 将两张图片转换为 NumPy 数组
image1_array = np.array(image1)
image2_array = np.array(image2)

# 创建一个空白的黑色底图
result = np.zeros((height, width, 4), dtype=np.uint8)

# 遍历图片的每个像素
for y in range(height):
    for x in range(width):
        # 获取两张图片当前像素的RGBA值
        pixel1 = image1_array[y, x]
        pixel2 = image2_array[y, x]

        # 判断两张图片的白色部分
        if np.all(pixel1[:3] == [255, 255, 255]) and np.all(pixel2[:3] == [255, 255, 255]):
            # 白色重叠部分显示为白色
            result[y, x] = [255, 255, 255, 255]
        elif np.all(pixel1[:3] == [255, 255, 255]):
            # 仅图片1中的白色部分显示为灰色
            result[y, x] = [169, 169, 169, 255]
        elif np.all(pixel2[:3] == [255, 255, 255]):
            # 仅图片2中的白色部分显示为灰色
            result[y, x] = [169, 169, 169, 255]
        else:
            # 其他区域保持黑色
            result[y, x] = [0, 0, 0, 255]

# 将结果数组转换回图片
result_image = Image.fromarray(result)

# 保存输出的图片
result_image.save('RDR_image.png')

# 显示结果图片
result_image.show()
