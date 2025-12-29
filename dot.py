from PIL import Image
import numpy as np

def reverse_transform(image_path, output_path):
    """
    实现图像的反转变换，并保存原始图像和反转后的图像。
    :param image_path: 输入图像的路径。
    :param output_path: 输出图像的路径。
    """
    # 打开图像并转换为灰度图
    image = Image.open(image_path).convert('L')
    # 将图像转换为numpy数组
    image_array = np.array(image)

    # 反转变换
    L = 255  # 8位图像的最大灰度级
    reversed_array = (L - 1) - image_array

    # 将处理后的数组转换回图像
    reversed_image = Image.fromarray(reversed_array.astype(np.uint8))

    # 保存原始图像和反转后的图像
    image.save('pics/original_image.jpg')
    reversed_image.save(output_path)
    print(f"Original image saved to original_image.jpg")
    print(f"Reverse transformed image saved to {output_path}")

def logarithmic_transform(image_path, output_path, c=1.0):
    """
    实现图像的对数变换。
    :param image_path: 输入图像的路径。
    :param output_path: 输出图像的路径。
    :param c: 对数变换中的常数，默认为1.0。
    """
    # 打开图像并转换为灰度图
    image = Image.open(image_path).convert('L')
    # 将图像转换为numpy数组
    image_array = np.array(image)

    # 对数变换
    logarithmic_array = c * np.log1p(image_array)

    # 归一化到[0, 255]范围
    logarithmic_array = (logarithmic_array / np.max(logarithmic_array) * 255).astype(np.uint8)

    # 将处理后的数组转换回图像
    logarithmic_image = Image.fromarray(logarithmic_array)
    logarithmic_image.save(output_path)
    print(f"Logarithmic transformed image saved to {output_path}")

# 使用示例
if __name__ == "__main__":
    input_image_path = 'pics/moulengdangubulei.jpg'  
    output_reverse_path = 'pics/reverse_transformed_image.jpg'
    output_logarithmic_path = 'pics/logarithmic_transformed_image.jpg'

    reverse_transform(input_image_path, output_reverse_path)
    logarithmic_transform(input_image_path, output_logarithmic_path)