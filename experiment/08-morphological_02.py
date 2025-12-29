import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from typing import Tuple, Optional, Dict, Any

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MorphologyProcessor:
    """形态学图像处理器基类"""
    
    def __init__(self):
        self.debug = False
    
    def check_image_file(self, image_path: str) -> bool:
        """检查图像文件是否存在且可读"""
        if not os.path.exists(image_path):
            print(f"错误：文件 '{image_path}' 不存在")
            return False
        
        if not os.path.isfile(image_path):
            print(f"错误：'{image_path}' 不是文件")
            return False
        
        return True
    
    def load_image(self, image_path: str, color_mode: str = 'gray') -> Optional[np.ndarray]:
        """
        加载图像
        
        参数:
        image_path: 图像路径
        color_mode: 'gray'灰度图, 'color'彩色图
        
        返回:
        图像数组或None
        """
        try:
            if not self.check_image_file(image_path):
                return None
            
            if color_mode == 'gray':
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(image_path)
                if img is not None and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img is None:
                print(f"错误：无法读取图像 '{image_path}'")
            
            return img
            
        except Exception as e:
            print(f"加载图像时出错: {e}")
            return None
    
    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        """保存图像到文件"""
        try:
            if image is None:
                print("错误：图像为空，无法保存")
                return False
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 如果是RGB图像，需要转换回BGR格式保存
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(output_path, image_bgr)
            else:
                success = cv2.imwrite(output_path, image)
            
            if success:
                print(f"图像已保存到: {output_path}")
            else:
                print(f"保存图像失败: {output_path}")
            
            return success
            
        except Exception as e:
            print(f"保存图像时出错: {e}")
            return False


class EdgeDetector(MorphologyProcessor):
    """边缘检测器"""
    
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    def extract_edges_morphological(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        使用形态学方法提取边缘
        
        参数:
        image: 输入图像
        
        返回:
        包含各种边缘结果的字典
        """
        results = {}
        
        # 确保图像是灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 方法1: 形态学梯度 (膨胀-腐蚀)
        dilated = cv2.dilate(gray, self.kernel, iterations=1)
        eroded = cv2.erode(gray, self.kernel, iterations=1)
        gradient = cv2.absdiff(dilated, eroded)
        results['gradient'] = gradient
        
        # 方法2: 外部梯度 (膨胀-原图)
        external = cv2.subtract(dilated, gray)
        results['external'] = external
        
        # 方法3: 内部梯度 (原图-腐蚀)
        internal = cv2.subtract(gray, eroded)
        results['internal'] = internal
        
        # 方法4: 拉普拉斯边缘检测 (对比)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        results['laplacian'] = laplacian
        
        # 方法5: Canny边缘检测 (对比)
        canny = cv2.Canny(gray, 100, 200)
        results['canny'] = canny
        
        # 二值化梯度边缘
        _, binary_gradient = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
        results['binary_gradient'] = binary_gradient
        
        results['original'] = gray
        
        return results
    
    def visualize_edges(self, results: Dict[str, np.ndarray], 
                        save_path: Optional[str] = None) -> None:
        """
        可视化边缘提取结果
        
        参数:
        results: 边缘结果字典
        save_path: 保存路径，如果为None则显示图像
        """
        titles = {
            'original': '原始图像',
            'gradient': '形态学梯度',
            'external': '外部梯度',
            'internal': '内部梯度',
            'laplacian': '拉普拉斯',
            'canny': 'Canny边缘',
            'binary_gradient': '二值化梯度'
        }
        
        # 选择要显示的图像
        display_keys = ['original', 'gradient', 'external', 'internal', 
                       'laplacian', 'canny', 'binary_gradient']
        display_keys = [k for k in display_keys if k in results]
        
        n_images = len(display_keys)
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1)
        
        for idx, key in enumerate(display_keys):
            ax = axes.flatten()[idx]
            img = results[key]
            ax.imshow(img, cmap='gray')
            ax.set_title(titles.get(key, key))
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(len(display_keys), len(axes.flatten())):
            axes.flatten()[idx].axis('off')
        
        plt.suptitle('边缘提取结果对比', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


class HoleFiller(MorphologyProcessor):
    """孔洞填充器"""
    
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    def preprocess_image(self, image: np.ndarray, threshold: int = 200, 
                        invert: bool = True) -> np.ndarray:
        """
        预处理图像：二值化
        
        参数:
        image: 输入图像
        threshold: 二值化阈值
        invert: 是否反转（True: 黑底白物体，False: 白底黑物体）
        
        返回:
        二值化图像
        """
        # 确保是灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if invert:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def find_holes(self, binary: np.ndarray) -> np.ndarray:
        """
        在二值图像中找到孔洞
        
        参数:
        binary: 二值图像（黑底白物体）
        
        返回:
        孔洞掩膜
        """
        # 寻找外部轮廓
        contours, hierarchy = cv2.findContours(
            binary, 
            cv2.RETR_CCOMP, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 创建孔洞掩膜
        holes = np.zeros_like(binary)
        
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                # 如果轮廓有父轮廓（即它是孔洞）
                if hierarchy[0][i][3] >= 0:
                    cv2.drawContours(holes, [contour], -1, 255, -1)
        
        return holes
    
    def find_seed_point(self, binary: np.ndarray) -> Tuple[int, int]:
        """
        自动寻找种子点
        
        参数:
        binary: 二值图像（黑底白物体）
        
        返回:
        种子点坐标 (x, y)
        """
        height, width = binary.shape
        
        # 方法1: 寻找孔洞位置
        holes = self.find_holes(binary)
        
        if np.any(holes > 0):
            # 取第一个孔洞的质心
            y_coords, x_coords = np.where(holes > 0)
            if len(x_coords) > 0 and len(y_coords) > 0:
                # 取中心附近的点
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                
                # 确保点在孔洞内
                if 0 <= center_x < width and 0 <= center_y < height and holes[center_y, center_x] > 0:
                    return (center_x, center_y)
        
        # 方法2: 如果找不到孔洞，尝试图像中心
        center_x, center_y = width // 2, height // 2
        
        # 如果中心点是前景，则寻找背景点
        if binary[center_y, center_x] > 0:
            # 寻找最近的背景点
            for radius in range(1, min(width, height) // 2):
                for angle in range(0, 360, 10):
                    rad = np.radians(angle)
                    x = int(center_x + radius * np.cos(rad))
                    y = int(center_y + radius * np.sin(rad))
                    
                    if 0 <= x < width and 0 <= y < height and binary[y, x] == 0:
                        return (x, y)
        
        # 方法3: 返回左上角
        return (10, 10)
    
    def fill_holes_morphological(self, binary: np.ndarray, 
                                seed_point: Optional[Tuple[int, int]] = None,
                                max_iterations: int = 500,
                                capture_intermediate: bool = True) -> Dict[str, Any]:
        """
        使用形态学方法填充孔洞
        
        参数:
        binary: 二值图像（黑底白物体）
        seed_point: 种子点，如果为None则自动寻找
        max_iterations: 最大迭代次数
        capture_intermediate: 是否捕获中间结果
        
        返回:
        包含填充结果和信息的字典
        """
        height, width = binary.shape
        
        # 1. 准备种子点
        if seed_point is None:
            seed_point = self.find_seed_point(binary)
        
        print(f"种子点: ({seed_point[0]}, {seed_point[1]})")
        
        # 2. 初始化标记图像
        marker = np.zeros((height, width), dtype=np.uint8)
        marker[seed_point[1], seed_point[0]] = 255
        
        # 3. 存储中间结果
        intermediate = {}
        convergence_iteration = -1
        
        # 4. 形态学重建
        for i in range(max_iterations):
            # 保存特定迭代次数的结果
            if capture_intermediate:
                if i == 0:
                    intermediate['iter_0'] = marker.copy()
                elif i == 10:
                    intermediate['iter_10'] = marker.copy()
                elif i == 50:
                    intermediate['iter_50'] = marker.copy()
                elif i == 100:
                    intermediate['iter_100'] = marker.copy()
            
            # 膨胀并与原图像取交集
            marker_dilated = cv2.dilate(marker, self.kernel, iterations=1)
            marker_new = cv2.bitwise_and(marker_dilated, binary)
            
            # 检查是否收敛
            if np.array_equal(marker_new, marker):
                convergence_iteration = i
                if self.debug:
                    print(f"收敛于第 {i} 次迭代")
                break
            
            marker = marker_new
            
            # 安全中断
            if i == max_iterations - 1:
                print(f"警告: 达到最大迭代次数 {max_iterations}")
                convergence_iteration = i
        
        # 5. 合并填充区域
        filled = cv2.bitwise_or(binary, marker)
        
        # 6. 计算填充的孔洞
        holes_filled = cv2.bitwise_xor(filled, binary)
        
        # 统计信息
        stats = {
            'convergence_iteration': convergence_iteration,
            'holes_pixels': np.sum(holes_filled > 0),
            'total_pixels': height * width,
            'holes_percentage': np.sum(holes_filled > 0) / (height * width) * 100
        }
        
        return {
            'filled': filled,
            'holes_filled': holes_filled,
            'marker': marker,
            'intermediate': intermediate,
            'seed_point': seed_point,
            'stats': stats
        }
    
    def fill_holes_contour(self, binary: np.ndarray) -> np.ndarray:
        """
        使用轮廓方法填充孔洞
        
        参数:
        binary: 二值图像
        
        返回:
        填充后的图像
        """
        # 复制图像
        filled = binary.copy()
        
        # 寻找轮廓（使用层级信息）
        contours, hierarchy = cv2.findContours(
            filled, 
            cv2.RETR_CCOMP,  # 获取所有轮廓及其层级关系
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if hierarchy is not None:
            # 填充所有孔洞
            for i, contour in enumerate(contours):
                # 检查是否为孔洞（有父轮廓）
                if hierarchy[0][i][3] >= 0:
                    cv2.drawContours(filled, [contour], -1, 255, -1)
        
        return filled
    
    def fill_holes_floodfill(self, binary: np.ndarray) -> np.ndarray:
        """
        使用泛洪填充方法填充孔洞
        
        参数:
        binary: 二值图像（白底黑物体）
        
        返回:
        填充后的图像
        """
        # 复制并反转图像（泛洪填充需要白底黑物体）
        if np.mean(binary) > 127:  # 如果主要是白色
            inverted = cv2.bitwise_not(binary)
        else:
            inverted = binary.copy()
        
        # 创建掩膜（比原图大2个像素）
        h, w = inverted.shape
        mask = np.zeros((h+2, w+2), np.uint8)
        
        # 从角落开始泛洪填充
        cv2.floodFill(inverted, mask, (0, 0), 255)
        
        # 反转回来
        filled = cv2.bitwise_not(inverted)
        
        return filled
    
    def visualize_hole_filling(self, binary: np.ndarray, results: Dict[str, Any],
                              method: str = "形态学填充", 
                              save_path: Optional[str] = None) -> None:
        """
        可视化孔洞填充结果
        
        参数:
        binary: 原始二值图像
        results: 填充结果字典
        method: 方法名称
        save_path: 保存路径
        """
        # 计算边缘
        dilated = cv2.dilate(binary, self.kernel, iterations=1)
        eroded = cv2.erode(binary, self.kernel, iterations=1)
        edges = cv2.absdiff(dilated, eroded)
        
        # 补集
        complement = cv2.bitwise_not(binary)
        
        # 获取填充结果
        filled = results.get('filled')
        holes_filled = results.get('holes_filled', np.zeros_like(binary))
        marker = results.get('marker', np.zeros_like(binary))
        intermediate = results.get('intermediate', {})
        
        # 准备显示图像
        display_images = []
        display_titles = []
        
        # 基础图像
        display_images.extend([binary, complement, edges])
        display_titles.extend(['原始二值图像', '原始补集', '边缘图像'])
        
        # 中间结果（如果有）
        if intermediate:
            if 'iter_0' in intermediate:
                display_images.append(intermediate['iter_0'])
                display_titles.append('初始种子')
            
            if 'iter_10' in intermediate:
                display_images.append(intermediate['iter_10'])
                display_titles.append('10次迭代')
            
            if 'iter_50' in intermediate:
                display_images.append(intermediate['iter_50'])
                display_titles.append('50次迭代')
            
            if 'iter_100' in intermediate:
                display_images.append(intermediate['iter_100'])
                display_titles.append('100次迭代')
        
        # 最终结果
        display_images.extend([marker, holes_filled, filled])
        display_titles.extend(['最终标记', '填充的孔洞', f'{method}结果'])
        
        # 创建子图
        n_images = len(display_images)
        n_cols = 3
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        
        # 展平axes数组以便遍历
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1)
        
        # 显示每个图像
        for idx, (img, title) in enumerate(zip(display_images, display_titles)):
            ax = axes.flatten()[idx]
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(len(display_images), len(axes.flatten())):
            axes.flatten()[idx].axis('off')
        
        # 添加统计信息
        stats = results.get('stats', {})
        if stats:
            info_text = f"收敛迭代: {stats.get('convergence_iteration', 'N/A')}\n"
            info_text += f"填充像素: {stats.get('holes_pixels', 0):,}\n"
            info_text += f"填充比例: {stats.get('holes_percentage', 0):.2f}%"
            
            fig.text(0.02, 0.98, info_text, fontsize=10, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'孔洞填充 - {method}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


class MorphologyToolkit:
    """形态学图像处理工具箱主类"""
    
    def __init__(self):
        self.edge_detector = EdgeDetector(kernel_size=3)
        self.hole_filler = HoleFiller(kernel_size=3)
        
    def process_edge_detection(self, input_path: str, 
                              output_dir: str = "./output",
                              save_results: bool = True) -> bool:
        """
        处理边缘提取
        
        参数:
        input_path: 输入图像路径
        output_dir: 输出目录
        save_results: 是否保存结果
        
        返回:
        成功与否
        """
        print(f"\n{'='*60}")
        print("开始边缘提取处理")
        print(f"{'='*60}")
        
        # 加载图像
        image = self.edge_detector.load_image(input_path, color_mode='gray')
        if image is None:
            return False
        
        print(f"图像尺寸: {image.shape}")
        print(f"图像类型: {image.dtype}")
        
        # 提取边缘
        results = self.edge_detector.extract_edges_morphological(image)
        
        # 保存结果
        if save_results:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存单个结果图像
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            for key, img in results.items():
                if key != 'original':  # 不保存原始图像
                    output_path = os.path.join(output_dir, f"{base_name}_{key}.png")
                    self.edge_detector.save_image(img, output_path)
            
            # 保存汇总图
            summary_path = os.path.join(output_dir, f"{base_name}_edge_summary.png")
            self.edge_detector.visualize_edges(results, save_path=summary_path)
        
        # 显示结果
        self.edge_detector.visualize_edges(results, save_path=None)
        
        print("边缘提取完成!")
        return True
    
    def process_hole_filling(self, input_path: str,
                            threshold: int = 200,
                            seed_point: Optional[Tuple[int, int]] = None,
                            method: str = "morphological",
                            output_dir: str = "./output",
                            save_results: bool = True) -> bool:
        """
        处理孔洞填充
        
        参数:
        input_path: 输入图像路径
        threshold: 二值化阈值
        seed_point: 种子点 (x, y)
        method: 填充方法 (morphological, contour, floodfill)
        output_dir: 输出目录
        save_results: 是否保存结果
        
        返回:
        成功与否
        """
        print(f"\n{'='*60}")
        print("开始孔洞填充处理")
        print(f"{'='*60}")
        
        # 加载图像
        image = self.hole_filler.load_image(input_path, color_mode='gray')
        if image is None:
            return False
        
        print(f"图像尺寸: {image.shape}")
        print(f"图像类型: {image.dtype}")
        
        # 预处理（二值化）
        binary = self.hole_filler.preprocess_image(image, threshold=threshold, invert=True)
        print(f"二值化阈值: {threshold}")
        
        # 根据选择的方法填充孔洞
        results = {}
        method_name = ""
        
        if method == "contour":
            print("使用轮廓方法填充孔洞...")
            filled = self.hole_filler.fill_holes_contour(binary)
            method_name = "轮廓填充"
            results = {'filled': filled}
            
        elif method == "floodfill":
            print("使用泛洪填充方法填充孔洞...")
            filled = self.hole_filler.fill_holes_floodfill(binary)
            method_name = "泛洪填充"
            results = {'filled': filled}
            
        else:  # 默认使用形态学方法
            print("使用形态学方法填充孔洞...")
            results = self.hole_filler.fill_holes_morphological(
                binary, 
                seed_point=seed_point,
                max_iterations=500,
                capture_intermediate=True
            )
            method_name = "形态学填充"
            
            # 打印统计信息
            stats = results.get('stats', {})
            if stats:
                print(f"收敛迭代次数: {stats.get('convergence_iteration', 'N/A')}")
                print(f"填充像素数: {stats.get('holes_pixels', 0):,}")
                print(f"填充比例: {stats.get('holes_percentage', 0):.2f}%")
        
        # 保存结果
        if save_results:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存结果图像
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            # 保存二值图像
            binary_path = os.path.join(output_dir, f"{base_name}_binary.png")
            self.hole_filler.save_image(binary, binary_path)
            
            # 保存填充结果
            filled = results.get('filled')
            if filled is not None:
                filled_path = os.path.join(output_dir, f"{base_name}_filled_{method}.png")
                self.hole_filler.save_image(filled, filled_path)
            
            # 保存汇总图
            summary_path = os.path.join(output_dir, f"{base_name}_hole_filling_{method}.png")
            self.hole_filler.visualize_hole_filling(
                binary, results, method=method_name, save_path=summary_path
            )
        
        # 显示结果
        self.hole_filler.visualize_hole_filling(
            binary, results, method=method_name, save_path=None
        )
        
        print("孔洞填充完成!")
        return True
    
    def compare_hole_filling_methods(self, input_path: str,
                                    threshold: int = 200,
                                    output_dir: str = "./output") -> bool:
        """
        比较不同的孔洞填充方法
        
        参数:
        input_path: 输入图像路径
        threshold: 二值化阈值
        output_dir: 输出目录
        
        返回:
        成功与否
        """
        print(f"\n{'='*60}")
        print("比较孔洞填充方法")
        print(f"{'='*60}")
        
        # 加载图像
        image = self.hole_filler.load_image(input_path, color_mode='gray')
        if image is None:
            return False
        
        # 预处理（二值化）
        binary = self.hole_filler.preprocess_image(image, threshold=threshold, invert=True)
        
        # 应用不同的填充方法
        methods = ['morphological', 'contour', 'floodfill']
        results = {}
        
        for method in methods:
            print(f"\n应用 {method} 方法...")
            
            if method == 'morphological':
                result = self.hole_filler.fill_holes_morphological(
                    binary, 
                    seed_point=None,
                    max_iterations=300,
                    capture_intermediate=False
                )
                results[method] = result.get('filled')
                
            elif method == 'contour':
                results[method] = self.hole_filler.fill_holes_contour(binary)
                
            elif method == 'floodfill':
                results[method] = self.hole_filler.fill_holes_floodfill(binary)
        
        # 可视化比较
        self.visualize_comparison(binary, results, output_dir)
        
        return True
    
    def visualize_comparison(self, binary: np.ndarray, 
                           results: Dict[str, np.ndarray],
                           output_dir: str) -> None:
        """
        可视化不同方法的比较结果
        
        参数:
        binary: 原始二值图像
        results: 不同方法的结果字典
        output_dir: 输出目录
        """
        # 计算边缘
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        eroded = cv2.erode(binary, kernel, iterations=1)
        edges = cv2.absdiff(dilated, eroded)
        
        # 方法名称映射
        method_names = {
            'morphological': '形态学方法',
            'contour': '轮廓方法',
            'floodfill': '泛洪填充'
        }
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 第一行：原始图像、边缘、补集
        row1_images = [binary, edges, cv2.bitwise_not(binary)]
        row1_titles = ['原始二值图像', '边缘图像', '补集']
        
        for idx, (img, title) in enumerate(zip(row1_images, row1_titles)):
            ax = axes[0, idx]
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        
        # 第二行：不同方法的填充结果
        methods = ['morphological', 'contour', 'floodfill']
        
        for idx, method in enumerate(methods):
            ax = axes[1, idx]
            
            if method in results:
                img = results[method]
                ax.imshow(img, cmap='gray')
                ax.set_title(method_names.get(method, method), fontsize=12)
                
                # 计算并显示差异
                diff = cv2.absdiff(img, binary)
                diff_pixels = np.sum(diff > 0)
                total_pixels = binary.shape[0] * binary.shape[1]
                percentage = diff_pixels / total_pixels * 100
                
                ax.text(0.5, -0.1, f"差异像素: {diff_pixels:,} ({percentage:.2f}%)", 
                       transform=ax.transAxes, ha='center', fontsize=10)
            else:
                ax.text(0.5, 0.5, f"{method} 结果不可用", 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.axis('off')
        
        plt.suptitle('孔洞填充方法比较', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "hole_filling_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"比较结果已保存到: {output_path}")
        
        plt.show()
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='形态学图像处理工具箱 - 边缘提取和孔洞填充',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --edge moon.jpg
  %(prog)s --hole moon1.jpg --threshold 200 --method morphological
  %(prog)s --edge moon.jpg --hole moon1.jpg --output ./results
  %(prog)s --compare moon1.jpg
        """
    )
    
    # 输入选项
    parser.add_argument('--edge', type=str, help='边缘提取输入图像路径')
    parser.add_argument('--hole', type=str, help='孔洞填充输入图像路径')
    parser.add_argument('--compare', type=str, help='比较孔洞填充方法，输入图像路径')
    
    # 处理参数
    parser.add_argument('--threshold', type=int, default=200,
                       help='二值化阈值 (默认: 200)')
    parser.add_argument('--method', type=str, default='morphological',
                       choices=['morphological', 'contour', 'floodfill'],
                       help='孔洞填充方法 (默认: morphological)')
    parser.add_argument('--seed-x', type=int, help='种子点X坐标')
    parser.add_argument('--seed-y', type=int, help='种子点Y坐标')
    
    # 输出选项
    parser.add_argument('--output', type=str, default='./output',
                       help='输出目录 (默认: ./output)')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果，仅显示')
    
    # 其他选项
    parser.add_argument('--kernel', type=int, default=3,
                       help='结构元素大小 (默认: 3)')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    args = parser.parse_args()
    
    # 检查是否有任何输入
    if not (args.edge or args.hole or args.compare):
        parser.print_help()
        print("\n错误: 必须指定至少一个处理选项 (--edge, --hole, 或 --compare)")
        sys.exit(1)
    
    # 创建工具箱
    toolkit = MorphologyToolkit()
    
    # 设置调试模式
    if args.debug:
        toolkit.edge_detector.debug = True
        toolkit.hole_filler.debug = True
    
    # 更新内核大小
    if args.kernel != 3:
        toolkit.edge_detector.kernel_size = args.kernel
        toolkit.edge_detector.kernel = np.ones((args.kernel, args.kernel), dtype=np.uint8)
        toolkit.hole_filler.kernel = np.ones((args.kernel, args.kernel), dtype=np.uint8)
    
    # 处理边缘提取
    if args.edge:
        success = toolkit.process_edge_detection(
            input_path=args.edge,
            output_dir=args.output,
            save_results=not args.no_save
        )
        if not success:
            print("边缘提取失败!")
    
    # 处理孔洞填充
    if args.hole:
        seed_point = None
        if args.seed_x is not None and args.seed_y is not None:
            seed_point = (args.seed_x, args.seed_y)
        
        success = toolkit.process_hole_filling(
            input_path=args.hole,
            threshold=args.threshold,
            seed_point=seed_point,
            method=args.method,
            output_dir=args.output,
            save_results=not args.no_save
        )
        if not success:
            print("孔洞填充失败!")
    
    # 比较孔洞填充方法
    if args.compare:
        success = toolkit.compare_hole_filling_methods(
            input_path=args.compare,
            threshold=args.threshold,
            output_dir=args.output
        )
        if not success:
            print("方法比较失败!")
    
    print("\n处理完成!")


if __name__ == "__main__":
    main()