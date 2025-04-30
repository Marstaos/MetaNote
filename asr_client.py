"""
ASR客户端模块
用于与ASR服务通信，处理音频文件的语音识别
"""

import os
import json
import logging
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ASRClient:
    """ASR客户端类"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        初始化ASR客户端
        
        Args:
            server_url: ASR服务器URL
        """
        self.server_url = server_url.rstrip('/')
    
    def check_health(self) -> bool:
        """
        检查ASR服务健康状态
        
        Returns:
            服务是否健康
        """
        try:
            response = requests.get(f"{self.server_url}/health")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"服务状态: {data['status']}")
                logger.info(f"模型已加载: {data['model_loaded']}")
                return data['model_loaded']
            else:
                logger.error(f"服务检查失败: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.error("无法连接到ASR服务，请确保服务正在运行")
            return False
        except Exception as e:
            logger.error(f"检查服务健康状态时出错: {str(e)}")
            return False
    
    def recognize_audio(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        调用ASR API进行语音识别
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            识别结果，失败时返回None
        """
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            logger.error(f"文件不存在: {audio_path}")
            return None
        
        # 准备文件
        files = {
            'file': (os.path.basename(audio_path), open(audio_path, 'rb'))
        }
        
        try:
            # 发送请求
            logger.info(f"正在识别音频文件: {audio_path}")
            response = requests.post(f"{self.server_url}/asr/recognize", files=files)
            
            # 关闭文件
            files['file'][1].close()
            
            # 处理响应
            if response.status_code == 200:
                result = response.json()
                logger.info(f"识别成功!")
                logger.info(f"文件名: {result['filename']}")
                logger.info(f"处理时间: {result.get('processing_time', 'N/A')}")
                return result
            else:
                logger.error(f"识别失败: {response.status_code}")
                logger.error(f"错误信息: {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            logger.error("无法连接到ASR服务，请确保服务正在运行")
            return None
        except Exception as e:
            logger.error(f"识别时发生错误: {str(e)}")
            # 确保文件句柄被关闭
            try:
                files['file'][1].close()
            except:
                pass
            return None
    
    def process_video(self, video_path: str, extract_audio_func=None) -> Optional[Dict[str, Any]]:
        """
        处理视频文件的音频
        
        Args:
            video_path: 视频文件路径
            extract_audio_func: 提取音频的函数，需要接受视频路径并返回音频路径
            
        Returns:
            识别结果，失败时返回None
        """
        if not extract_audio_func:
            logger.error("未提供提取音频函数")
            return None
            
        try:
            # 提取音频
            audio_path = extract_audio_func(video_path)
            if not audio_path:
                logger.error("从视频提取音频失败")
                return None
                
            # 识别音频
            result = self.recognize_audio(audio_path)
            
            # 清理临时音频文件
            if os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception as e:
                    logger.warning(f"清理临时音频文件失败: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"处理视频时出错: {str(e)}")
            return None

    def recognize_multiple_files(self, audio_files: List[str]) -> List[Dict[str, Any]]:
        """
        批量识别多个音频文件
        
        Args:
            audio_files: 音频文件路径列表
        
        Returns:
            识别结果列表
        """
        results = []
        for audio_file in audio_files:
            logger.info(f"\n正在处理文件: {audio_file}")
            result = self.recognize_audio(audio_file)
            if result:
                results.append({
                    "file": audio_file,
                    "result": result
                })
        return results

# 便捷函数
def recognize_audio(audio_path: str, server_url: str = "http://localhost:8000") -> Optional[Dict[str, Any]]:
    """
    识别音频文件的便捷函数
    
    Args:
        audio_path: 音频文件路径
        server_url: ASR服务器URL
        
    Returns:
        识别结果，失败时返回None
    """
    client = ASRClient(server_url)
    return client.recognize_audio(audio_path)


if __name__ == "__main__":
    import argparse
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ASR客户端")
    parser.add_argument("--server", default="http://localhost:8000", help="ASR服务器URL")
    parser.add_argument("--file", help="要识别的音频文件路径")
    parser.add_argument("--check", action="store_true", help="检查服务健康状态")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = ASRClient(args.server)
    
    # 检查服务
    if args.check:
        health = client.check_health()
        print(f"服务健康状态: {'正常' if health else '异常'}")
    
    # 识别文件
    if args.file:
        if not os.path.exists(args.file):
            print(f"文件不存在: {args.file}")
            exit(1)
            
        result = client.recognize_audio(args.file)
        if result:
            print("\n识别结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("识别失败")