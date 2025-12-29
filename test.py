import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import os

# 创建输出目录
output_dir = "output_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

# 图片反转
img = cv2.imread("yueqiu.jfif",0)
img_neg = 255 - img

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('原始图片')
plt.subplot(122), plt.imshow(img_neg, cmap='gray'), plt.title('反转变换')
plt.show()
cv2.imwrite(os.path.join(output_dir,"negetive_transform.jpg"),img_neg)

# 对数变换
def log_transform(img, c=1):
    img_normalized = img / 255.0  # 归一化到[0,1]
    s = c * np.log(1 + img_normalized)
    return np.uint8(s * 255)

# 应用对数变换
img_log = log_transform(img, c=50)
cv2.imwrite(os.path.join(output_dir, "log_transform.jpg"), img_log)
cv2.imshow('Log Transform', img_log)
cv2.waitKey(0)


def gamma_correction(img, gamma=1.0, c=1):
    table = c * (np.arange(256) / 255.0) ** gamma * 255
    table = np.clip(table, 0, 255).astype('uint8')
    return cv2.LUT(img, table)


# 不同gamma值对比
# gammas = [0.4, 0.6, 1.5]
gammas = [2, 3, 4]
results = [gamma_correction(img, g) for g in gammas]

# 保存伽马校正结果
for i, gamma in enumerate(gammas):
    cv2.imwrite(os.path.join(output_dir, f"gamma_{gamma}.jpg"), results[i])

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(results[i], cmap='gray')
    plt.title(f'γ={gammas[i]}')
plt.savefig(os.path.join(output_dir, "gamma_comparison.jpg"))
plt.show()