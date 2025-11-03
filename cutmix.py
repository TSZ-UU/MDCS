from PIL import Image, ImageDraw, ImageColor
import random

def cutmix(a_img_path, b_img_path, paste_size=(100, 100), box_color="red"):
    # 打开两张图片
    a_img = Image.open(a_img_path)
    b_img = Image.open(b_img_path)

    # 获取两张图片的尺寸
    a_width, a_height = a_img.size
    b_width, b_height = b_img.size

    # 确保粘贴区域大小不大于a图像的尺寸
    paste_width, paste_height = paste_size
    paste_width = min(paste_width, a_width)
    paste_height = min(paste_height, a_height)

    # 随机选择A图像中的一个粘贴区域的位置
    x1 = random.randint(0, a_width - paste_width)
    y1 = random.randint(0, a_height - paste_height)
    x2 = x1 + paste_width
    y2 = y1 + paste_height

    # 从A图像中裁剪出区域
    paste_region = a_img.crop((x1, y1, x2, y2))

    # 在B图像上随机选择一个粘贴位置（确保与A的裁剪区域对应）
    x_offset = x1  # 确保x坐标对应
    y_offset = y1  # 确保y坐标对应

    # 将A图像的区域粘贴到B图像
    b_img.paste(paste_region, (x_offset, y_offset))

    # 在B图像上绘制一个提示框
    draw = ImageDraw.Draw(b_img)
    # 使用ImageColor.getrgb将RGB元组转换为整数格式
    if isinstance(box_color, tuple) and len(box_color) == 1:
        color_int = ImageColor.getrgb(f'rgb{box_color}')
        draw.rectangle([x_offset, y_offset, x_offset + paste_width, y_offset + paste_height], outline=color_int, width=10)
    else:
        print("Error: box_color must be an RGB tuple, e.g. (255, 0, 0)")

    return b_img


# 示例调用
a_img_path = r"D:\Project\SynFoC-master\code\generate_images\Xw_BIDMC_00_00.png"
b_img_path = r"D:\Project\SynFoC-master\code\generate_images\00_06.png"
result_img = cutmix(a_img_path, b_img_path, paste_size=(150, 200))

# 显示结果
result_img.show()

# 可以保存结果
result_img.save("result_image.jpg")
