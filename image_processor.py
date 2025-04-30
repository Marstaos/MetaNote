"""
图像理解处理模块
支持使用Ollama本地模型或阿里云千问多模态API进行图像内容理解
"""

import os
import base64
import json
import logging
import time
import threading
import queue
from typing import Dict, Any, List, Optional, Union
import requests
import cv2
from concurrent.futures import ThreadPoolExecutor

# 尝试导入OpenAI库 (用于千问API)
try:
    from openai import OpenAI
    has_openai = True
except ImportError:
    has_openai = False

logger = logging.getLogger(__name__)

class ImageProcessor:
    """图像理解处理基类"""
    
    def __init__(self):
        """初始化图像理解处理器"""
        pass
    
    def process_image(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        处理单个图像
        
        Args:
            image_path: 图像文件路径
            prompt: 提示语
            
        Returns:
            处理结果
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def process_images(self, image_paths: List[str], prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量处理多个图像
        
        Args:
            image_paths: 图像文件路径列表
            prompt: 提示语
            
        Returns:
            处理结果列表
        """
        results = []
        for path in image_paths:
            try:
                result = self.process_image(path, prompt)
                results.append({
                    'path': path,
                    'result': result
                })
            except Exception as e:
                logger.error(f"处理图像 {path} 时出错: {str(e)}")
                results.append({
                    'path': path,
                    'result': {
                        'is_valuable': False,
                        'description': "",
                        'score': 0,
                        'error': str(e)
                    }
                })
        return results
    
    def process_frame_extractor(self, frame_extractor, prompt: Optional[str] = None) -> None:
        """
        处理FrameExtractor中的所有图像
        
        Args:
            frame_extractor: FrameExtractor实例
            prompt: 提示语
        """
        # 获取所有帧
        frames = frame_extractor.get_all_frames()
        
        # 准备图像路径列表
        image_paths_map = {}
        for frame in frames:
            image_path = os.path.join(frame_extractor.frames_dir, frame['filename'])
            if os.path.exists(image_path):
                image_paths_map[image_path] = frame['id']
        
        # 处理所有图像
        results = self.process_images(list(image_paths_map.keys()), prompt)
        
        # 更新FrameExtractor中的帧信息
        processed_count = 0
        for item in results:
            path = item['path']
            result = item['result']
            if path in image_paths_map:
                frame_id = image_paths_map[path]
                # 更新FrameExtractor中的帧评估结果
                frame_extractor.update_frame_value(
                    frame_id,
                    result.get('is_valuable', False),
                    result.get('description', "")
                )
                processed_count += 1
        
        logger.info(f"已处理并更新 {processed_count} 个帧")
    
    def get_default_prompt(self) -> str:
        """
        获取默认的提示语
        
        Returns:
            默认提示语
        """
        return (
            "这是一个教学视频中的帧。请评估这个图像是否包含有价值的教学内容"
            "（如幻灯片、图表、代码、公式、重要概念、关键要点），判断是否合适作为笔记的配图。"
            "请简要分析理由，并在回答末尾用[是/否]明确标记你的判断。"
        )
    
    def _is_image_file(self, file_path: str) -> bool:
        """
        检查文件是否为图像
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为图像文件
        """
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        ext = os.path.splitext(file_path)[1].lower()
        return ext in valid_extensions


class OllamaProcessor(ImageProcessor):
    """使用本地Ollama多模态模型进行图像理解"""
    
    def __init__(
        self, 
        url: str = "http://localhost:11434/api/generate",
        model: str = "llava:13b",
        parallel_requests: int = 2
    ):
        """
        初始化Ollama处理器
        
        Args:
            url: Ollama API地址
            model: 使用的模型名称
            parallel_requests: 并行请求数量
        """
        super().__init__()
        self.url = url
        self.model = model
        self.parallel_requests = parallel_requests
        self.initialized = False
    
    def initialize(self) -> bool:
        """
        初始化连接并测试服务可用性
        
        Returns:
            初始化是否成功
        """
        try:
            # 尝试进行一次简单请求以检查服务是否可用
            payload = {
                "model": self.model,
                "prompt": "Hello, are you ready?",
                "stream": False
            }
            
            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Ollama服务 ({self.url}) 初始化成功，使用模型: {self.model}")
                self.initialized = True
                return True
            else:
                logger.error(f"Ollama服务初始化失败: {response.status_code}")
                logger.error(f"返回内容: {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error(f"无法连接到Ollama服务: {self.url}")
            return False
        except Exception as e:
            logger.error(f"初始化Ollama服务时出错: {str(e)}")
            return False
    
    def process_image(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        使用Ollama处理单个图像
        
        Args:
            image_path: 图像文件路径
            prompt: 提示语
            
        Returns:
            处理结果
        """
        if not self.initialized and not self.initialize():
            raise RuntimeError("Ollama服务未初始化或不可用")
            
        if not self._is_image_file(image_path):
            logger.warning(f"不是有效的图像文件: {image_path}")
            return {'is_valuable': False, 'description': "", 'score': 0, 'reason': "不是有效的图像文件"}
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 转换为base64格式
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 使用提供的prompt或默认prompt
            if prompt is None:
                prompt = self.get_default_prompt()
            
            # 构造请求体
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }
            
            # 发送请求
            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=60)
            
            # 处理返回
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # 解析结果
                is_valuable = '[是]' in response_text or response_text.strip().endswith('是')
                score = 75 if is_valuable else 25
                
                # 提取描述（移除判断部分）
                description = response_text
                if '[是]' in description:
                    description = description.replace('[是]', '').strip()
                elif '[否]' in description:
                    description = description.replace('[否]', '').strip()
                elif description.strip().endswith('是'):
                    description = description[:-1].strip()
                elif description.strip().endswith('否'):
                    description = description[:-1].strip()
                
                return {
                    'is_valuable': is_valuable,
                    'description': description,
                    'score': score,
                    'reason': response_text
                }
            else:
                logger.error(f"Ollama请求失败: {response.status_code}")
                logger.error(f"返回内容: {response.text}")
                return {
                    'is_valuable': False,
                    'description': "",
                    'score': 0,
                    'reason': f"API请求失败: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"处理图像时出错: {str(e)}")
            return {
                'is_valuable': False,
                'description': "",
                'score': 0,
                'reason': f"处理出错: {str(e)}"
            }
    
    def process_images(self, image_paths: List[str], prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量处理多个图像
        
        Args:
            image_paths: 图像文件路径列表
            prompt: 提示语
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.parallel_requests) as executor:
            # 提交所有任务
            future_to_path = {
                executor.submit(self.process_image, path, prompt): path 
                for path in image_paths if self._is_image_file(path)
            }
            
            # 收集结果
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append({
                        'path': path,
                        'result': result
                    })
                except Exception as e:
                    logger.error(f"处理图像 {path} 时出错: {str(e)}")
                    results.append({
                        'path': path,
                        'result': {
                            'is_valuable': False,
                            'description': "",
                            'score': 0,
                            'reason': f"处理出错: {str(e)}"
                        }
                    })
        
        return results


class QwenProcessor(ImageProcessor):
    """使用阿里云千问多模态API进行图像理解"""
    
    def __init__(
        self, 
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-vl-plus",
        max_workers: int = 2
    ):
        """
        初始化千问处理器
        
        Args:
            api_key: 千问API密钥
            base_url: API基础URL
            model: 使用的模型
            max_workers: 并行工作线程数量
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_workers = max_workers
        self.client = None
        self.initialized = False
        
        if not has_openai:
            logger.error("OpenAI库未安装，无法使用千问API。请安装: pip install openai")
    
    def initialize(self) -> bool:
        """
        初始化千问客户端
        
        Returns:
            初始化是否成功
        """
        if not has_openai:
            logger.error("OpenAI库未安装，无法使用千问API")
            return False
            
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # 发送简单请求验证连接
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Hello, are you ready?"}
                ]
            )
            
            if completion and hasattr(completion, 'choices'):
                logger.info(f"千问API ({self.base_url}) 初始化成功，使用模型: {self.model}")
                self.initialized = True
                return True
            else:
                logger.error("千问API响应异常")
                return False
                
        except Exception as e:
            logger.error(f"初始化千问API时出错: {str(e)}")
            return False
    
    def _encode_image(self, image_path: str) -> str:
        """
        将图像编码为base64格式
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            base64编码的图像
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def process_image(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        使用千问处理单个图像
        
        Args:
            image_path: 图像文件路径
            prompt: 提示语
            
        Returns:
            处理结果
        """
        if not has_openai:
            return {'is_valuable': False, 'description': "", 'score': 0, 'reason': "OpenAI库未安装，无法使用千问API"}
            
        if not self.initialized and not self.initialize():
            raise RuntimeError("千问API未初始化或不可用")
            
        if not self._is_image_file(image_path):
            logger.warning(f"不是有效的图像文件: {image_path}")
            return {'is_valuable': False, 'description': "", 'score': 0, 'reason': "不是有效的图像文件"}
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        try:
            # 使用提供的prompt或默认prompt
            if prompt is None:
                prompt = self.get_default_prompt()
            
            # 编码图像
            image_b64 = self._encode_image(image_path)
            
            # 构建请求
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]}
                ]
            )
            
            # 处理响应
            response_text = completion.choices[0].message.content
            
            # 解析结果
            is_valuable = '[是]' in response_text or response_text.strip().endswith('是')
            
            # 提取描述（移除判断部分）
            description = response_text
            if '[是]' in description:
                description = description.replace('[是]', '').strip()
            elif '[否]' in description:
                description = description.replace('[否]', '').strip()
            elif description.strip().endswith('是'):
                description = description[:-1].strip()
            elif description.strip().endswith('否'):
                description = description[:-1].strip()
            
            return {
                'is_valuable': is_valuable,
                'description': description,
                'score': 75 if is_valuable else 25,
                'reason': response_text
            }
            
        except Exception as e:
            logger.error(f"千问API请求失败: {str(e)}")
            return {
                'is_valuable': False,
                'description': "",
                'score': 0,
                'reason': f"API请求失败: {str(e)}"
            }
    
    def process_images(self, image_paths: List[str], prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量处理多个图像
        
        Args:
            image_paths: 图像文件路径列表
            prompt: 提示语
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_path = {
                executor.submit(self.process_image, path, prompt): path 
                for path in image_paths if self._is_image_file(path)
            }
            
            # 收集结果
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append({
                        'path': path,
                        'result': result
                    })
                except Exception as e:
                    logger.error(f"处理图像 {path} 时出错: {str(e)}")
                    results.append({
                        'path': path,
                        'result': {
                            'is_valuable': False,
                            'description': "",
                            'score': 0,
                            'reason': f"处理出错: {str(e)}"
                        }
                    })
        
        return results


def create_image_processor(config: Dict[str, Any]) -> ImageProcessor:
    """
    创建图像处理器
    
    Args:
        config: 配置字典
        
    Returns:
        图像处理器实例
    """
    provider = config.get('image_understanding', {}).get('provider', 'ollama')
    
    if provider.lower() == 'ollama':
        # 创建Ollama处理器
        ollama_config = config.get('image_understanding', {}).get('ollama', {})
        url = ollama_config.get('url', 'http://localhost:11434/api/generate')
        model = ollama_config.get('model', 'llava:13b')
        parallel_requests = config.get('video', {}).get('parallel_requests', 2)
        
        processor = OllamaProcessor(url, model, parallel_requests)
        
    elif provider.lower() == 'qwen':
        # 创建千问处理器
        qwen_config = config.get('image_understanding', {}).get('qwen', {})
        api_key = qwen_config.get('api_key', '')
        base_url = qwen_config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        model = qwen_config.get('model', 'qwen-vl-plus')
        parallel_requests = config.get('video', {}).get('parallel_requests', 2)
        
        processor = QwenProcessor(api_key, base_url, model, parallel_requests)
        
    else:
        raise ValueError(f"不支持的图像处理器类型: {provider}")
    
    # 初始化处理器
    success = processor.initialize()
    if not success:
        logger.warning(f"图像处理器 ({provider}) 初始化失败")
    
    return processor


if __name__ == "__main__":
    import argparse
    from utils import load_config
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='图像理解测试')
    parser.add_argument('--image', required=True, help='要测试的图像文件路径')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--provider', choices=['ollama', 'qwen'], help='要使用的处理器')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖处理器类型（如果指定）
    if args.provider:
        config['image_understanding'] = config.get('image_understanding', {})
        config['image_understanding']['provider'] = args.provider
    
    # 创建处理器
    processor = create_image_processor(config)
    
    # 处理图像
    result = processor.process_image(args.image)
    
    # 打印结果
    print("\n处理结果:")
    print(f"图像路径: {args.image}")
    print(f"有价值: {'是' if result.get('is_valuable', False) else '否'}")
    print(f"分数: {result.get('score', 0)}")
    print(f"描述: {result.get('description', '')}")
    print(f"完整理由: {result.get('reason', '')}")