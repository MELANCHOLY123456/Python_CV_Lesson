import cv2
import matplotlib.pyplot as plt
import numpy as np

def rgb2gray(rgb):
    """
        将RGB图像转换为灰度图像
        Args:
            rgb: 输入的三通道RGB图像
        Returns:
            gray: 输出的单通道灰度图像
        """
    # 使用标准的亮度转换公式：Y = 0.299R + 0.587G + 0.114B
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return gray

def thre_bin(gray_image, threshold=170):
    """
        对灰度图像进行二值化处理
        Args:
            gray_image: 输入的灰度图像
            threshold: 二值化阈值，默认为170
        Returns:
            threshold_image: 二值化图像，像素值为0或1
        """
    threshold_image = np.zeros(shape=(gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image[i][j] > threshold:
                threshold_image[i][j] = 1
            else:
                threshold_image[i][j] = 0
    return threshold_image

def erode_bin_image(bin_image, kernel):
    """
       对二值图像进行腐蚀操作
       Args:
           bin_image: 输入的二值图像，像素值为0或1
           kernel: 结构元素
       Returns:
           eroded_image: 腐蚀后的二值图像
       """
    kernel_size = kernel.shape[0]
    bin_image = np.array(bin_image)

    # 参数检查
    if(kernel_size%2 == 0) or kernel_size<1:
        raise ValueError("kernel size must be odd or bigger than 1")
    if(bin_image.max() != 1) or (bin_image.min() != 0):
        raise ValueError("input image's pixel value must be 0 or 1")

    # 创建输出图像
    d_image = np.zeros(shape=bin_image.shape)
    center_move = int((kernel_size-1)/2)

    for i in range(center_move, bin_image.shape[0]-kernel_size+1):
        for j in range(center_move, bin_image.shape[1]-kernel_size+1):
            d_image[i, j] = np.min(bin_image[i-center_move:i+center_move,
                                  j-center_move:j+center_move])
    return d_image

def dilate_bin_image(bin_image, kernel):
    """
        对二值图像进行膨胀操作
        Args:
            bin_image: 输入的二值图像，像素值为0或1
            kernel: 结构元素
        Returns:
            dilated_image: 膨胀后的二值图像
        """
    kernel_size = kernel.shape[0]
    bin_image = np.array(bin_image)
    if (kernel_size % 2 == 0) or kernel_size < 1:
        raise ValueError("kernel size must be odd or bigger than 1")
    if (bin_image.max() != 1) or (bin_image.min() != 0):
        raise ValueError("input image's pixel value must be 0 or 1")

    d_image = np.zeros(shape=bin_image.shape)
    center_move = int((kernel_size - 1)/2)

    for i in range(center_move, bin_image.shape[0]-kernel_size+1):
        for j in range(center_move, bin_image.shape[1]-kernel_size+1):
            d_image[i, j] = np.max(bin_image[i-center_move:i+center_move,
                                   j-center_move:j+center_move])
    return d_image

if __name__ == "__main__":
    # 1: 读取图像
    # 使用cv2读取图像，OpenCV默认读取为BGR格式
    image = np.array(cv2.imread("/home/haoboyang/Python_CV_Lesson/pics/moulengdangubulei.jpg")[:, :, 0:3])
    # 将BGR转换为RGB格式以便正确显示
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2: 转换为灰度图像
    image_gray = rgb2gray(image)

    # 3: 图像反转（暗区变亮，亮区变暗）
    invert_image = 255 - image_gray

    # 4: 二值化
    bin_image = thre_bin(invert_image)

    # 5: 定义结构元素并进行形态学操作
    # 使用5x5的结构元素进行腐蚀
    kernel = np.ones(shape=(5, 5))
    eroded_image = erode_bin_image(bin_image, kernel)
    # 使用13x13的结构元素进行膨胀（较大的核可以更明显地看到膨胀效果）
    kernel = np.ones(shape=(13, 13))
    dilated_image = dilate_bin_image(bin_image, kernel)

    # 6: 单独显示二值化图像（可选）
    plt.imshow(bin_image, cmap="gray")
    plt.show()

    # 7: 显示完整的图像处理流程
    plt.figure(figsize=(15, 3))

    # 原始灰度图像
    # 选择1行5列网格中的第1个位置
    plt.subplot(1, 5, 1)

    # 在当前选中的子图位置显示灰度图像
    # image_gray: 要显示的图像数据
    # cmap="gray": 颜色映射为灰度模式
    plt.imshow(image_gray, cmap="gray")

    # 为当前子图添加标题
    plt.title("Original")

    # 关闭坐标轴显示，让图片更干净
    plt.axis('off')

    # 反转图像
    plt.subplot(1, 5, 2)
    plt.imshow(invert_image, cmap="gray")
    plt.title("Inverted")
    plt.axis('off')

    # 二值化图像
    plt.subplot(1, 5, 3)
    plt.imshow(bin_image, cmap="gray")
    plt.title("Binary")
    plt.axis('off')

    # 腐蚀后的图像
    plt.subplot(1, 5, 4)
    plt.imshow(eroded_image, cmap="gray")
    plt.title("Eroded")
    plt.axis('off')

    # 膨胀后的图像
    plt.subplot(1, 5, 5)
    plt.imshow(dilated_image, cmap="gray")
    plt.title("Dilated")
    plt.axis('off')

    plt.tight_layout()
    plt.show()