import cv2
import numpy as np
from matplotlib import pyplot as plt

def sys_equalizehist(img):
    img = cv2.imread(img, 0)
    h, w = img.shape
    equ_img = cv2.equalizeHist(img)
    equ_hist = cv2.calcHist([equ_img], [0], None, [256], [0, 255])
    equ_hist[0:255] = equ_hist[0:255]/(h*w)
    # res = np.hstack((img,equ)) #stacking images side-by-side#这一行是将两个图像进行了行方向的叠加
    return [equ_img, equ_hist]

def def_equalizehist(img,L=256):
    img = cv2.imread(img,0)

    h,w = img.shape
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    hist[0:255] = hist[0:255]/(h*w)

    sum_hist = np.zeros(hist.shape)
    for i in range(256):
        sum_hist[i] = np.sum(hist[0:i+1])

    equal_hist = np.zeros(sum_hist.shape, dtype=np.uint8)
    for i in range(256):
        equal_hist[i] = int(((L-1)-0)*float(sum_hist[i]))

    equal_img = img.copy()
    for i in range(h):
        for j in range(w):
            equal_img[i, j] = equal_hist[int(img[i, j])]

    equal_hist = cv2.calcHist([equal_img], [0], None, [256], [0, 255])
    equal_hist[0:255] = equal_hist[0:255]/(h*w)
    return [equal_img,equal_hist]

if __name__=='__main__':
    img = "../pics/moulengdangubulei.jpg"
    sys_img, sys_hist = sys_equalizehist(img)
    def_img, def_hist = def_equalizehist(img)
    x = np.linspace(0, 255, 256)  # 创建x轴坐标数据
    plt.subplot(1, 2, 1), plt.plot(x, sys_hist, '-b')
    plt.subplot(1, 2, 2), plt.plot(x, def_hist, '-r')
    plt.show()

    ori_img = cv2.imread(img, 0)
    '''
    cv2.imshow('ori_img',ori_img)
    cv2.imshow('sys_img',sys_img)
    cv2.imshow('def_img',def_img)
    '''

    combined_img = np.hstack((ori_img, sys_img, def_img))
    cv2.imshow('Comparison: Original | System | Custom', combined_img)
    cv2.resizeWindow('Comparison: Original | System | Custom', 900, 300)  # 调整窗口大小
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Comparison: Original | System | Custom', cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()