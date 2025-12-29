'''
霍夫曼编码实现
'''

class HuffmanNode:
    """哈夫曼树节点类"""
    def __init__(self, char=None, freq=0):
        self.char = char      # 字符
        self.freq = freq      # 频率
        self.left = None      # 左子节点
        self.right = None     # 右子节点
    
    def __lt__(self, other):
        # 用于节点排序比较
        return self.freq < other.freq

class HuffmanTree:
    """哈夫曼树类"""
    def __init__(self, char_freq):
        """
        初始化哈夫曼树
        char_freq: 字符频率列表，格式为[(char1, freq1), (char2, freq2), ...]
        """
        # 创建初始叶子节点
        self.leaves = [HuffmanNode(char, freq) for char, freq in char_freq]
        self.build_tree()
    
    def build_tree(self):
        """构建哈夫曼树"""
        import heapq
        
        # 使用堆（优先队列）来高效获取最小频率节点
        heap = self.leaves[:]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            # 取出两个最小频率的节点
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # 创建新内部节点
            parent = HuffmanNode(freq=left.freq + right.freq)
            parent.left = left
            parent.right = right
            
            # 将新节点加入堆中
            heapq.heappush(heap, parent)
        
        # 根节点
        self.root = heap[0] if heap else None
    
    def generate_codes(self):
        """生成哈夫曼编码"""
        def traverse(node, current_code, codes):
            if node is None:
                return
            
            # 如果是叶子节点，记录编码
            if node.char is not None:
                codes[node.char] = current_code
                return
            
            # 遍历左子树和右子树
            traverse(node.left, current_code + '0', codes)
            traverse(node.right, current_code + '1', codes)
        
        codes = {}
        if self.root:
            traverse(self.root, '', codes)
        return codes
    
    def print_codes(self):
        """打印哈夫曼编码表"""
        codes = self.generate_codes()
        print("哈夫曼编码表:")
        print("-" * 30)
        for char, code in sorted(codes.items()):
            print(f"字符 '{char}' : {code}")
        print("-" * 30)
        return codes

class LZWEncoder:
    """LZW压缩编码器"""
    
    def __init__(self, initial_dict=None):
        """
        初始化LZW编码器
        initial_dict: 初始字典，默认为ASCII字符集
        """
        if initial_dict is None:
            # 初始化包含ASCII字符的字典
            self.dictionary = {}
            # 添加所有可打印ASCII字符
            for i in range(256):
                self.dictionary[chr(i)] = i
            self.next_code = 256
        else:
            self.dictionary = initial_dict.copy()
            self.next_code = max(initial_dict.values()) + 1 if initial_dict else 0
    
    def encode(self, text):
        """对文本进行LZW编码"""
        if not text:
            return []
        
        result = []
        current_string = ""
        
        for char in text:
            # 构建新的字符串
            new_string = current_string + char
            
            # 如果新字符串在字典中，继续扩展
            if new_string in self.dictionary:
                current_string = new_string
            else:
                # 输出当前字符串的编码
                result.append(self.dictionary[current_string])
                
                # 将新字符串加入字典
                self.dictionary[new_string] = self.next_code
                self.next_code += 1
                
                # 重置当前字符串为当前字符
                current_string = char
        
        # 输出最后一个字符串的编码
        if current_string:
            result.append(self.dictionary[current_string])
        
        return result
    
    def get_dictionary_size(self):
        """获取当前字典大小"""
        return len(self.dictionary)

def demo_huffman():
    """演示哈夫曼编码"""
    print("=" * 50)
    print("哈夫曼编码演示")
    print("=" * 50)
    
    # 测试数据
    char_weights = [
        ('a', 6), ('b', 4), ('c', 10), 
        ('d', 8), ('f', 12), ('g', 2)
    ]
    
    print(f"字符频率: {char_weights}")
    
    # 创建哈夫曼树并生成编码
    huffman_tree = HuffmanTree(char_weights)
    codes = huffman_tree.print_codes()
    
    # 计算编码效率
    total_chars = sum(freq for _, freq in char_weights)
    avg_length = sum(len(codes[char]) * freq for char, freq in char_weights) / total_chars
    print(f"平均编码长度: {avg_length:.2f}")
    
    return codes

def demo_lzw():
    """演示LZW编码"""
    print("\n" + "=" * 50)
    print("LZW编码演示")
    print("=" * 50)
    
    # 测试文本
    test_string = "thisisthe"
    print(f"原始文本: '{test_string}'")
    
    # 创建LZW编码器
    encoder = LZWEncoder()
    
    # 进行编码
    encoded = encoder.encode(test_string)
    print(f"LZW编码结果: {encoded}")
    print(f"编码后字典大小: {encoder.get_dictionary_size()}")
    
    # 计算压缩率
    original_bits = len(test_string) * 8  # 假设原始每个字符8位
    encoded_bits = len(encoded) * 12      # 假设编码后每个码12位（LZW常用）
    compression_ratio = (1 - encoded_bits / original_bits) * 100
    print(f"压缩率: {compression_ratio:.1f}%")
    
    return encoded

def main():
    """主函数"""
    print("压缩算法演示程序")
    print("作者: AI助手")
    print()
    
    # 演示哈夫曼编码
    huffman_codes = demo_huffman()
    
    # 演示LZW编码
    lzw_encoded = demo_lzw()
    
    # 综合示例：使用哈夫曼编码对LZW结果进行编码
    print("\n" + "=" * 50)
    print("综合示例: 对LZW编码结果进行哈夫曼编码")
    print("=" * 50)
    
    # 统计LZW编码结果的频率
    from collections import Counter
    lzw_freq = Counter(lzw_encoded)
    lzw_char_freq = [(str(code), freq) for code, freq in lzw_freq.items()]
    
    print(f"LZW编码频率分布: {lzw_char_freq}")
    
    # 对LZW编码结果进行哈夫曼编码
    if lzw_char_freq:
        lzw_huffman = HuffmanTree(lzw_char_freq)
        lzw_huffman_codes = lzw_huffman.print_codes()
        
        # 编码LZW结果
        encoded_lzw = []
        for code in lzw_encoded:
            encoded_lzw.append(lzw_huffman_codes[str(code)])
        
        print(f"哈夫曼编码后的LZW结果: {encoded_lzw}")
    
    print("\n演示结束!")

if __name__ == '__main__':
    main()