import cv2
import numpy as np
import os

def robert_operator(img):
    """
    Robert算子边缘检测
    使用标准的Robert算子核进行边缘检测
    
    参数:
        img: 输入灰度图像
    
    返回:
        边缘检测结果图像
    """
    if len(img.shape) != 2:
        raise ValueError("输入图像必须为灰度图")
    
    r, c = img.shape
    result = np.zeros((r, c), dtype=np.float32)
    
    # 标准Robert算子核
    kernel_x = np.array([[-1, 0], [0, 1]])
    kernel_y = np.array([[0, -1], [1, 0]])
    
    # 边界处理：只处理内部像素
    for i in range(r - 1):
        for j in range(c - 1):
            region = img[i:i + 2, j:j + 2].astype(np.float32)
            gx = np.sum(region * kernel_x)
            gy = np.sum(region * kernel_y)
            result[i, j] = np.sqrt(gx ** 2 + gy ** 2)
    
    return np.uint8(result)

def sobel_operator(img):
    """
    Sobel算子边缘检测
    使用Sobel算子进行边缘检测，包含X和Y方向
    
    参数:
        img: 输入灰度图像
    
    返回:
        边缘检测结果图像
    """
    if len(img.shape) != 2:
        raise ValueError("输入图像必须为灰度图")
    
    r, c = img.shape
    result = np.zeros((r, c), dtype=np.float32)
    
    # Sobel算子核
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # 边界处理：只处理内部像素，周围一圈保持原值或0
    for i in range(1, r - 1):
        for j in range(1, c - 1):
            region = img[i-1:i+2, j-1:j+2].astype(np.float32)
            gx = np.sum(region * kernel_x)
            gy = np.sum(region * kernel_y)
            result[i, j] = np.sqrt(gx ** 2 + gy ** 2)
    
    return np.uint8(result)

def laplace_operator(img):
    """
    Laplace算子边缘检测
    使用Laplace算子进行二阶导数边缘检测
    
    参数:
        img: 输入灰度图像
    
    返回:
        边缘检测结果图像
    """
    if len(img.shape) != 2:
        raise ValueError("输入图像必须为灰度图")
    
    r, c = img.shape
    result = np.zeros((r, c), dtype=np.float32)
    
    # Laplace算子核（4邻域）
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    
    # 边界处理：只处理内部像素
    for i in range(1, r - 1):
        for j in range(1, c - 1):
            region = img[i-1:i+2, j-1:j+2].astype(np.float32)
            result[i, j] = abs(np.sum(region * kernel))
    
    return np.uint8(result)

def unsharp_mask(img, strength=1.5, radius=1):
    """
    Unsharp Mask锐化算法
    通过模糊图像与原图的差异实现锐化
    
    参数:
        img: 输入图像
        strength: 锐化强度
        radius: 模糊半径
    
    返回:
        锐化后的图像
    """
    # 高斯模糊
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    
    # 计算差异
    mask = cv2.subtract(img, blurred)
    
    # 应用锐化
    sharpened = cv2.addWeighted(img, 1.0, mask, strength, 0)
    
    return np.uint8(np.clip(sharpened, 0, 255))

def sharpen_image(original, edges, strength=0.5):
    """
    基于边缘检测的图像锐化
    
    参数:
        original: 原始图像
        edges: 边缘检测结果
        strength: 锐化强度(0.0-1.0)
    
    返回:
        锐化后的图像
    """
    # 参数验证
    if strength < 0 or strength > 2:
        raise ValueError("锐化强度应在0-2范围内")
    
    # 转换为float32避免溢出
    original_float = original.astype(np.float32)
    edges_float = edges.astype(np.float32)
    
    # 锐化：原图 + 强度 × 边缘
    sharpened = original_float + strength * edges_float
    
    # 限制到有效范围
    sharpened = np.clip(sharpened, 0, 255)
    
    return np.uint8(sharpened)

def create_comparison_display(original, edges_list, sharpened_list, operator_names):
    """
    创建对比显示图像
    
    参数:
        original: 原始图像
        edges_list: 边缘检测结果列表
        sharpened_list: 锐化结果列表
        operator_names: 算子名称列表
    
    返回:
        拼接后的对比图像
    """
    # 调整图像大小以便显示
    target_height = 200
    original_resized = cv2.resize(original, (int(original.shape[1] * target_height / original.shape[0]), target_height))
    
    # 创建第一行：边缘检测结果
    edge_images = [original_resized]
    for edges in edges_list:
        edges_resized = cv2.resize(edges, (original_resized.shape[1], target_height))
        edge_images.append(edges_resized)
    
    # 创建第二行：锐化结果
    sharp_images = [original_resized]
    for sharpened in sharpened_list:
        sharp_resized = cv2.resize(sharpened, (original_resized.shape[1], target_height))
        sharp_images.append(sharp_resized)
    
    # 水平拼接
    top_row = np.hstack(edge_images)
    bottom_row = np.hstack(sharp_images)
    
    # 垂直拼接
    combined = np.vstack((top_row, bottom_row))
    
    return combined

def main():
    """主函数"""
    # 使用项目中的实际图像路径
    image_path = "/home/haoboyang/Python_CV_Lesson/pics/moulengdangubulei.jpg"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在 - {image_path}")
        return
    
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("错误: 无法读取图像文件")
        return
    
    print(f"图像尺寸: {img.shape}")
    print("正在执行边缘检测和锐化处理...")
    
    try:
        # 边缘检测
        edges_robert = robert_operator(img)
        edges_sobel = sobel_operator(img)
        edges_laplace = laplace_operator(img)
        
        # 锐化处理
        sharp_robert = sharpen_image(img, edges_robert, strength=0.6)
        sharp_sobel = sharpen_image(img, edges_sobel, strength=0.3)
        sharp_laplace = sharpen_image(img, edges_laplace, strength=0.2)
        
        # Unsharp Mask锐化
        sharp_unsharp = unsharp_mask(img, strength=1.5, radius=1.0)
        
        # 创建对比显示
        operator_names = ['Original', 'Robert', 'Sobel', 'Laplace']
        edges_list = [edges_robert, edges_sobel, edges_laplace]
        sharpened_list = [sharp_robert, sharp_sobel, sharp_laplace]
        
        # 添加Unsharp Mask到对比
        edges_list.append(np.zeros_like(img))  # 占位符
        sharpened_list.append(sharp_unsharp)
        operator_names.append('Unsharp')
        
        # 创建显示图像
        comparison_image = create_comparison_display(img, edges_list, sharpened_list, operator_names)
        
        # 显示结果
        cv2.namedWindow('Sharpening Filter Comparison', cv2.WINDOW_NORMAL)
        cv2.imshow('Sharpening Filter Comparison', comparison_image)
        
        # 调整窗口大小
        h, w = comparison_image.shape
        cv2.resizeWindow('Sharpening Filter Comparison', w, h)
        
        print("\n显示布局说明:")
        print("上排: 原始图像 | Robert边缘 | Sobel边缘 | Laplace边缘 | 占位符")
        print("下排: 原始图像 | Robert锐化 | Sobel锐化 | Laplace锐化 | Unsharp锐化")
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
                output_path = "experiment/sharpening_comparison.jpg"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, comparison_image)
                print(f"对比图像已保存到: {output_path}")
            elif cv2.getWindowProperty('Sharpening Filter Comparison', cv2.WND_PROP_VISIBLE) < 1:
                break
    
    except Exception as e:
        print(f"处理过程中出错: {e}")
    
    finally:
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()