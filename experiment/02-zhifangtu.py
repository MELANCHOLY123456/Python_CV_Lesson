import cv2
import numpy as np
from matplotlib import pyplot as plt

def sys_equalizehist(img):
    img = cv2.imread(img, 0)
    h, w = img.shape
    equ_img = cv2.equalizeHist(img)
    equ_hist = cv2.calcHist([equ_img], [0], None, [256], [0, 255])
    equ_hist = equ_hist.flatten()  # 转换为1D数组
    equ_hist = equ_hist / (h * w)  # 归一化
    return [equ_img, equ_hist]

def def_equalizehist(img, L=256):
    img = cv2.imread(img, 0)
    h, w = img.shape
    
    # 计算直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    hist = hist.flatten()  # 转换为1D数组
    hist = hist / (h * w)  # 归一化
    
    # 计算累积分布函数
    sum_hist = np.zeros(256)
    for i in range(256):
        sum_hist[i] = np.sum(hist[0:i+1])
    
    # 计算均衡化映射
    equal_hist = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        equal_hist[i] = int((L-1) * sum_hist[i])
    
    # 应用均衡化
    equal_img = img.copy()
    for i in range(h):
        for j in range(w):
            equal_img[i, j] = equal_hist[img[i, j]]
    
    # 计算均衡化后的直方图
    equal_hist_result = cv2.calcHist([equal_img], [0], None, [256], [0, 255])
    equal_hist_result = equal_hist_result.flatten()
    equal_hist_result = equal_hist_result / (h * w)
    
    return [equal_img, equal_hist_result]

# 添加图像缩放函数
def resize_image(image, max_width=800):
    """将图像缩放到指定最大宽度，保持宽高比"""
    h, w = image.shape
    if w <= max_width:
        return image
    
    # 计算缩放比例
    scale = max_width / w
    new_width = max_width
    new_height = int(h * scale)
    
    # 使用interpolation参数确保图像质量
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

if __name__=='__main__':
    img_path = "/home/haoboyang/Python_CV_Lesson/pics/moulengdangubulei.jpg"
    sys_img, sys_hist = sys_equalizehist(img_path)
    def_img, def_hist = def_equalizehist(img_path)
    
    # 绘制直方图
    x = np.linspace(0, 255, 256)
    plt.figure(figsize=(10, 4))  # 设置图形大小
    plt.subplot(1, 2, 1)
    plt.plot(x, sys_hist, '-b')
    plt.title('System Equalization Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.plot(x, def_hist, '-r')
    plt.title('Custom Equalization Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()  # 自动调整子图布局
    plt.show()
    
    # 读取原始图像
    ori_img = cv2.imread(img_path, 0)
    
    # 将三个图像水平堆叠
    combined_img = np.hstack((ori_img, sys_img, def_img))
    
    # 缩放图像到合适大小
    combined_img_resized = resize_image(combined_img, max_width=1200)
    
    # 显示图像
    cv2.namedWindow('Comparison: Original | System | Custom', cv2.WINDOW_NORMAL)
    cv2.imshow('Comparison: Original | System | Custom', combined_img_resized)
    
    # 获取缩放后图像的大小并调整窗口大小
    h, w = combined_img_resized.shape
    cv2.resizeWindow('Comparison: Original | System | Custom', w, h)
    
    # print(f"原始组合图像尺寸: {combined_img.shape}")
    # print(f"缩放后图像尺寸: {combined_img_resized.shape}")
    # print("按 'q' 键退出窗口")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty('Comparison: Original | System | Custom', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()