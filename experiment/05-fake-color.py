import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def get_r_component(gray):
    """
    计算伪彩色的红色分量
    
    参数:
        gray: 灰度值 (0-255)
    
    返回:
        红色分量 (0-255)
    """
    # 使用int16避免溢出
    r = np.zeros_like(gray, dtype=np.int16)
    
    # 灰度值 <= 127: R = 0
    # 灰度值 > 191: R = 255  
    # 127 < 灰度值 <= 191: R = (gray-127)*4-1
    mask1 = gray <= 127
    mask2 = gray > 191
    mask3 = ~mask1 & ~mask2
    
    r[mask2] = 255
    r[mask3] = (gray[mask3] - 127) * 4 - 1
    
    # 确保值在0-255范围内
    r = np.clip(r, 0, 255).astype(np.uint8)
    
    return r

def get_g_component(gray):
    """
    计算伪彩色的绿色分量
    
    参数:
        gray: 灰度值 (0-255)
    
    返回:
        绿色分量 (0-255)
    """
    g = np.zeros_like(gray, dtype=np.uint8)
    
    # 灰度值 < 64: G = 4*gray
    # 灰度值 > 191: G = 256-(gray-191)*4
    # 64 <= 灰度值 <= 191: G = 255
    mask1 = gray < 64
    mask2 = gray > 191
    mask3 = ~mask1 & ~mask2
    
    # 先计算再限制范围，避免溢出
    g_temp = np.zeros_like(gray, dtype=np.int16)  # 使用int16避免溢出
    g_temp[mask1] = 4 * gray[mask1]
    g_temp[mask2] = 255 - (gray[mask2] - 192) * 4  # 修正公式：256-(gray-191)*4 = 255-(gray-192)*4
    g_temp[mask3] = 255
    
    # 确保值在0-255范围内
    g = np.clip(g_temp, 0, 255).astype(np.uint8)
    
    return g

def get_b_component(gray):
    """
    计算伪彩色的蓝色分量
    
    参数:
        gray: 灰度值 (0-255)
    
    返回:
        蓝色分量 (0-255)
    """
    # 使用int16避免溢出
    b = np.zeros_like(gray, dtype=np.int16)
    
    # 灰度值 < 64: B = 255
    # 灰度值 > 127: B = 0
    # 64 <= 灰度值 <= 127: B = 255-(gray-64)*4
    mask1 = gray < 64
    mask2 = gray > 127
    mask3 = ~mask1 & ~mask2
    
    b[mask1] = 255
    b[mask2] = 0
    b[mask3] = 255 - (gray[mask3] - 64) * 4
    
    # 确保值在0-255范围内
    b = np.clip(b, 0, 255).astype(np.uint8)
    
    return b

def apply_continuous_pseudo_color(gray_image):
    """
    应用连续伪彩色变换
    
    参数:
        gray_image: 输入灰度图像
    
    返回:
        伪彩色图像
    """
    if len(gray_image.shape) != 2:
        raise ValueError("输入图像必须为灰度图")
    
    # 使用向量化操作计算RGB分量
    r = get_r_component(gray_image)
    g = get_g_component(gray_image)
    b = get_b_component(gray_image)
    
    # 合并为BGR图像（OpenCV格式）
    pseudo_color = cv2.merge([b, g, r])
    
    return pseudo_color

def apply_discrete_pseudo_color(gray_image, color_array):
    """
    应用离散伪彩色变换
    
    参数:
        gray_image: 输入灰度图像
        color_array: 颜色映射数组
    
    返回:
        伪彩色图像
    """
    if len(gray_image.shape) != 2:
        raise ValueError("输入图像必须为灰度图")
    
    h, w = gray_image.shape
    pseudo_color = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 将灰度值映射到颜色数组索引
    indices = (gray_image // 16).astype(int)
    
    # 应用颜色映射
    for i in range(h):
        for j in range(w):
            pseudo_color[i, j] = color_array[indices[i, j]]
    
    return pseudo_color

def create_jet_colormap():
    """
    创建Jet颜色映射（类似MATLAB的jet colormap）
    
    返回:
        256个颜色的Jet颜色映射数组
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    
    for i in range(256):
        if i < 32:
            colormap[i] = [0, 0, i * 8]
        elif i < 96:
            colormap[i] = [0, (i - 32) * 4, 255]
        elif i < 160:
            colormap[i] = [(i - 96) * 4, 255, 255 - (i - 96) * 4]
        elif i < 224:
            colormap[i] = [255, 255 - (i - 160) * 4, 0]
        else:
            colormap[i] = [255 - (i - 224) * 4, 0, 0]
    
    return colormap

def apply_jet_pseudo_color(gray_image):
    """
    应用Jet伪彩色变换
    
    参数:
        gray_image: 输入灰度图像
    
    返回:
        Jet伪彩色图像
    """
    if len(gray_image.shape) != 2:
        raise ValueError("输入图像必须为灰度图")
    
    jet_colormap = create_jet_colormap()
    h, w = gray_image.shape
    pseudo_color = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 应用Jet颜色映射
    for i in range(h):
        for j in range(w):
            pseudo_color[i, j] = jet_colormap[gray_image[i, j]]
    
    return pseudo_color

def create_comparison_display(original, pseudo_images, titles):
    """
    创建对比显示图像
    
    参数:
        original: 原始灰度图像
        pseudo_images: 伪彩色图像列表
        titles: 标题列表
    
    返回:
        拼接后的对比图像
    """
    # 调整图像大小以便显示
    target_height = 300
    original_resized = cv2.resize(original, (int(original.shape[1] * target_height / original.shape[0]), target_height))
    original_color = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
    
    # 创建图像列表
    images = [original_color]
    for pseudo in pseudo_images:
        pseudo_resized = cv2.resize(pseudo, (original_resized.shape[1], target_height))
        images.append(pseudo_resized)
    
    # 水平拼接
    combined = np.hstack(images)
    
    return combined

def plot_colormap_comparison(colormaps, titles):
    """
    绘制颜色映射对比图
    
    参数:
        colormaps: 颜色映射数组列表
        titles: 标题列表
    """
    plt.figure(figsize=(12, 3))
    
    for i in range(len(colormaps)):
        plt.subplot(1, len(colormaps), i + 1)
        
        # 创建渐变条
        colored_gradient = np.zeros((256, 20, 3), dtype=np.uint8)
        for j in range(256):
            colored_gradient[j] = colormaps[i][j]
        
        plt.imshow(colored_gradient)
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    # 使用项目中的实际图像路径
    image_path = "/home/haoboyang/Python_CV_Lesson/pics/moulengdangubulei.jpg"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在 - {image_path}")
        return
    
    # 读取图像
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        print("错误: 无法读取图像文件")
        return
    
    print(f"图像尺寸: {gray_image.shape}")
    print("正在执行伪彩色处理...")
    
    # 预定义的离散颜色数组
    fc_array = [
        (0, 51, 0), (0, 51, 102), (51, 51, 102), (51, 102, 51),
        (51, 51, 153), (102, 51, 102), (153, 153, 0), (51, 102, 153),
        (153, 102, 51), (153, 204, 102), (204, 153, 102), (102, 204, 102),
        (153, 204, 153), (204, 204, 102), (204, 255, 204), (255, 255, 204)
    ]
    
    try:
        # 应用不同的伪彩色变换
        continuous_pseudo = apply_continuous_pseudo_color(gray_image)
        discrete_pseudo = apply_discrete_pseudo_color(gray_image, fc_array)
        jet_pseudo = apply_jet_pseudo_color(gray_image)
        
        # 创建对比显示
        pseudo_images = [continuous_pseudo, discrete_pseudo, jet_pseudo]
        titles = ['Original', 'Continuous', 'Discrete', 'Jet']
        comparison_image = create_comparison_display(gray_image, pseudo_images, titles)
        
        # 显示结果
        cv2.namedWindow('Pseudo Color Comparison', cv2.WINDOW_NORMAL)
        cv2.imshow('Pseudo Color Comparison', comparison_image)
        
        # 调整窗口大小 - 修复：使用shape的前两个维度
        h, w = comparison_image.shape[:2]
        cv2.resizeWindow('Pseudo Color Comparison', w, h)
        
        # 创建颜色映射对比
        jet_colormap = create_jet_colormap()
        continuous_colormap = []
        
        # 创建连续伪彩色的颜色映射
        for i in range(256):
            r = get_r_component(np.array([[i]]))[0, 0]
            g = get_g_component(np.array([[i]]))[0, 0]
            b = get_b_component(np.array([[i]]))[0, 0]
            continuous_colormap.append([b, g, r])  # BGR格式
        
        # 转换为numpy数组
        continuous_colormap = np.array(continuous_colormap, dtype=np.uint8)
        
        # 扩展离散颜色映射到256个颜色
        extended_discrete = []
        for i in range(256):
            extended_discrete.append(fc_array[i // 16])
        extended_discrete = np.array(extended_discrete, dtype=np.uint8)
        
        # 显示颜色映射对比
        plot_colormap_comparison([continuous_colormap, extended_discrete, jet_colormap], 
                               ['Continuous', 'Discrete', 'Jet'])
        
        print("\n显示布局说明:")
        print("原始图像 | 连续伪彩色 | 离散伪彩色 | Jet伪彩色")
        print("\n操作说明:")
        print("- 按 'q' 键退出程序")
        print("- 按 's' 键保存对比图像")
        print("- 关闭窗口也会退出程序")
        
        # 交互式操作
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存结果
                output_path = "experiment/pseudo_color_comparison.jpg"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, comparison_image)
                print(f"对比图像已保存到: {output_path}")
            elif cv2.getWindowProperty('Pseudo Color Comparison', cv2.WND_PROP_VISIBLE) < 1:
                break
    
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()