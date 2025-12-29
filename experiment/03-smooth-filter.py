import numpy as np
import cv2
import os

def means_filter(input_image, filter_size):
    """
    均值滤波函数
    
    参数:
        input_image: 输入图像(灰度图)
        filter_size: 滤波器尺寸(必须为奇数)
    
    返回:
        滤波后的图像
    """
    # 参数验证
    if filter_size <= 0 or filter_size % 2 == 0:
        raise ValueError("滤波器尺寸必须为正奇数")
    
    if len(input_image.shape) != 2:
        raise ValueError("输入图像必须为灰度图")
    
    input_image_cp = np.copy(input_image).astype(np.float32)
    filter_template = np.ones((filter_size, filter_size)) / (filter_size ** 2)
    pad_num = int((filter_size - 1) / 2)
    
    # 边界填充 - 使用反射填充比零填充效果更好
    input_image_padded = np.pad(input_image_cp, pad_num, mode="reflect")
    m, n = input_image_padded.shape
    output_image = np.zeros_like(input_image_cp)
    
    # 卷积操作
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            window = input_image_padded[i - pad_num:i + pad_num + 1, 
                                      j - pad_num:j + pad_num + 1]
            output_image[i - pad_num, j - pad_num] = np.sum(filter_template * window)
    
    return output_image.astype(np.uint8)

def gaussian_filter(input_image, filter_size, sigma=1.0):
    """
    高斯滤波函数
    
    参数:
        input_image: 输入图像(灰度图)
        filter_size: 滤波器尺寸(必须为奇数)
        sigma: 高斯分布的标准差
    
    返回:
        滤波后的图像
    """
    # 参数验证
    if filter_size <= 0 or filter_size % 2 == 0:
        raise ValueError("滤波器尺寸必须为正奇数")
    
    if len(input_image.shape) != 2:
        raise ValueError("输入图像必须为灰度图")
    
    # 创建高斯核
    center = filter_size // 2
    kernel = np.zeros((filter_size, filter_size))
    
    for i in range(filter_size):
        for j in range(filter_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # 归一化
    kernel = kernel / np.sum(kernel)
    
    # 应用滤波
    return cv2.filter2D(input_image, -1, kernel)

def compare_filters(image_path, filter_size=9):
    """
    比较不同滤波器的效果
    
    参数:
        image_path: 图像路径
        filter_size: 滤波器尺寸
    """
    # 读取图像
    original_image = cv2.imread(image_path, 0)
    if original_image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    
    print(f"原始图像尺寸: {original_image.shape}")
    print(f"使用滤波器尺寸: {filter_size}x{filter_size}")
    
    # 应用不同的平滑滤波
    try:
        # 自定义均值滤波
        mean_filtered = means_filter(original_image, filter_size)
        
        # OpenCV内置均值滤波
        cv_mean_filtered = cv2.blur(original_image, (filter_size, filter_size))
        
        # 高斯滤波
        gaussian_filtered = gaussian_filter(original_image, filter_size, sigma=filter_size/3)
        
        # OpenCV内置高斯滤波
        cv_gaussian_filtered = cv2.GaussianBlur(original_image, (filter_size, filter_size), 
                                               sigmaX=filter_size/3)
        
        # 创建对比图像
        top_row = np.hstack((original_image, mean_filtered, cv_mean_filtered))
        bottom_row = np.hstack((gaussian_filtered, cv_gaussian_filtered, 
                               np.ones_like(original_image) * 128))  # 占位符
        
        combined_image = np.vstack((top_row, bottom_row))
        
        # 调整图像大小以便显示
        max_width = 1200
        h, w = combined_image.shape
        if w > max_width:
            scale = max_width / w
            new_width = max_width
            new_height = int(h * scale)
            combined_image = cv2.resize(combined_image, (new_width, new_height))
        
        # 显示结果
        cv2.namedWindow('Smooth Filter Comparison', cv2.WINDOW_NORMAL)
        cv2.imshow('Smooth Filter Comparison', combined_image)
        
        # 添加标签
        print("\n图像布局说明:")
        print("上排: 原始图像 | 自定义均值滤波 | OpenCV均值滤波")
        print("下排: 自定义高斯滤波 | OpenCV高斯滤波 | 占位符")
        
        return combined_image
        
    except Exception as e:
        print(f"滤波处理出错: {e}")
        return None

def main():
    """主函数"""
    # 图像路径和参数
    image_path = "/home/haoboyang/Python_CV_Lesson/pics/moulengdangubulei.jpg"
    filter_size = 9
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在 - {image_path}")
        return
    
    try:
        # 比较不同滤波器效果
        result_image = compare_filters(image_path, filter_size)
        
        if result_image is not None:
            print("\n操作说明:")
            print("- 按 'q' 键退出程序")
            print("- 按 's' 键保存对比图像")
            print("- 关闭窗口也会退出程序")
            
            # 等待用户按键
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存结果
                    output_path = "experiment/smooth_filter_comparison.jpg"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, result_image)
                    print(f"对比图像已保存到: {output_path}")
                elif cv2.getWindowProperty('Smooth Filter Comparison', cv2.WND_PROP_VISIBLE) < 1:
                    break
        
    except Exception as e:
        print(f"程序运行出错: {e}")
    
    finally:
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()