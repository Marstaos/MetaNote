"""
工具函数模块
包含文件操作、视频处理等通用工具函数
"""

import os
import yaml
import logging
import tempfile
import json
import subprocess
import time
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import cv2

# 配置日志
logger = logging.getLogger(__name__)

# ===== 配置管理函数 =====

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}，将使用默认配置")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"已加载配置: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置出错: {str(e)}")
        return {}

def save_config(config: Dict[str, Any], config_path: str = "config.yaml") -> bool:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
        
    Returns:
        是否成功保存
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"配置已保存到: {config_path}")
        return True
    except Exception as e:
        logger.error(f"保存配置出错: {str(e)}")
        return False

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    从配置中获取值，支持点号路径
    
    Args:
        config: 配置字典
        key_path: 键路径，如 "asr.server_url"
        default: 默认值
        
    Returns:
        配置值
    """
    if not key_path:
        return default
    
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """
    设置配置值，支持点号路径
    
    Args:
        config: 配置字典
        key_path: 键路径，如 "asr.server_url"
        value: 要设置的值
        
    Returns:
        更新后的配置字典
    """
    if not key_path:
        return config
    
    keys = key_path.split('.')
    current = config
    
    # 遍历路径直到倒数第二个键
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # 设置最后一个键的值
    current[keys[-1]] = value
    
    return config

# ===== 目录和文件操作 =====

def ensure_directory(directory: str) -> str:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
        
    Returns:
        目录的绝对路径
    """
    abs_dir = os.path.abspath(directory)
    if not os.path.exists(abs_dir):
        os.makedirs(abs_dir, exist_ok=True)
        logger.debug(f"创建目录: {abs_dir}")
    return abs_dir

def is_video_file(file_path: str) -> bool:
    """
    检查是否为视频文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否为视频文件
    """
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    ext = os.path.splitext(file_path)[1].lower()
    return ext in valid_extensions

def is_audio_file(file_path: str) -> bool:
    """
    检查是否为音频文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否为音频文件
    """
    valid_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac', '.wma']
    ext = os.path.splitext(file_path)[1].lower()
    return ext in valid_extensions

def is_image_file(file_path: str) -> bool:
    """
    检查是否为图像文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否为图像文件
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
    ext = os.path.splitext(file_path)[1].lower()
    return ext in valid_extensions

# ===== 视频处理函数 =====

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    获取视频基本信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        包含视频信息的字典
    """
    if not os.path.exists(video_path):
        raise ValueError(f"视频文件不存在: {video_path}")
    
    # 使用OpenCV获取基本信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取基本属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    # 尝试使用ffprobe获取更详细信息
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            # 解析JSON结果
            ffprobe_info = json.loads(result.stdout)
            
            # 提取格式信息
            format_info = ffprobe_info.get("format", {})
            format_name = format_info.get("format_name", "unknown")
            bit_rate = format_info.get("bit_rate", "unknown")
            size = format_info.get("size", "unknown")
            
            # 提取视频流信息
            video_stream = None
            audio_stream = None
            
            for stream in ffprobe_info.get("streams", []):
                if stream.get("codec_type") == "video" and not video_stream:
                    video_stream = stream
                elif stream.get("codec_type") == "audio" and not audio_stream:
                    audio_stream = stream
            
            # 添加更多信息
            if video_stream:
                video_codec = video_stream.get("codec_name", "unknown")
            else:
                video_codec = "unknown"
                
            if audio_stream:
                audio_codec = audio_stream.get("codec_name", "unknown")
                audio_channels = audio_stream.get("channels", 0)
                audio_sample_rate = audio_stream.get("sample_rate", "unknown")
            else:
                audio_codec = "unknown"
                audio_channels = 0
                audio_sample_rate = "unknown"
            
            return {
                "filename": os.path.basename(video_path),
                "path": video_path,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "duration_formatted": format_duration(duration),
                "format": format_name,
                "size": size,
                "bit_rate": bit_rate,
                "video_codec": video_codec,
                "audio_codec": audio_codec,
                "audio_channels": audio_channels,
                "audio_sample_rate": audio_sample_rate
            }
    except Exception as e:
        logger.warning(f"使用ffprobe获取视频信息失败: {str(e)}")
    
    # 返回基本信息
    return {
        "filename": os.path.basename(video_path),
        "path": video_path,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "duration_formatted": format_duration(duration)
    }

def extract_audio(video_path: str, output_path: Optional[str] = None, format: str = "wav") -> str:
    """
    从视频中提取音频
    
    Args:
        video_path: 视频文件路径
        output_path: 输出音频文件路径，如果为None则生成临时文件
        format: 输出音频格式
        
    Returns:
        输出的音频文件路径
    """
    if not os.path.exists(video_path):
        raise ValueError(f"视频文件不存在: {video_path}")
    
    # 如果未指定输出路径，创建临时文件
    if not output_path:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}")
        temp_file.close()
        output_path = temp_file.name
    
    try:
        # 构建ffmpeg命令
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # 不处理视频
            "-ar", "16000",  # 设置采样率
            "-ac", "1",  # 设置声道数
            "-q:a", "0",  # 最高音质
            output_path,
            "-y"  # 覆盖已有文件
        ]
        
        logger.info(f"从视频提取音频: {video_path} -> {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"提取音频失败: {result.stderr}")
            raise RuntimeError(f"提取音频失败: {result.stderr}")
        
        logger.info(f"音频提取成功: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"提取音频时出错: {str(e)}")
        # 如果是临时文件且出错，清理它
        if not output_path and os.path.exists(output_path):
            os.unlink(output_path)
        raise

def capture_frame(video_path: str, timestamp: float, output_path: Optional[str] = None) -> str:
    """
    从视频指定时间点截取帧
    
    Args:
        video_path: 视频文件路径
        timestamp: 时间点（秒）
        output_path: 输出图像路径，如果为None则生成临时文件
        
    Returns:
        输出的图像文件路径
    """
    if not os.path.exists(video_path):
        raise ValueError(f"视频文件不存在: {video_path}")
    
    # 如果未指定输出路径，创建临时文件
    if not output_path:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.close()
        output_path = temp_file.name
    
    try:
        # 构建ffmpeg命令
        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),  # 设置时间点
            "-i", video_path,
            "-vframes", "1",  # 只提取一帧
            "-q:v", "2",  # 高质量
            output_path,
            "-y"  # 覆盖已有文件
        ]
        
        logger.info(f"从视频截取帧: {video_path}@{timestamp}s -> {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"截取帧失败: {result.stderr}")
            raise RuntimeError(f"截取帧失败: {result.stderr}")
        
        logger.info(f"帧截取成功: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"截取帧时出错: {str(e)}")
        # 如果是临时文件且出错，清理它
        if not output_path and os.path.exists(output_path):
            os.unlink(output_path)
        raise

def format_duration(seconds: float) -> str:
    """
    格式化持续时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

# ===== 初始化日志 =====

def setup_logging(config: Dict[str, Any]) -> None:
    """
    设置日志系统
    
    Args:
        config: 配置字典
    """
    log_config = config.get('logging', {})
    log_level_name = log_config.get('level', 'INFO')
    log_file = log_config.get('file', 'metanote.log')
    
    # 转换日志级别
    log_level = getattr(logging, log_level_name)
    
    # 创建处理器
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        
        # 添加文件处理器
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info(f"日志系统初始化完成，级别: {log_level_name}, 文件: {log_file}")

# ===== 全局初始化函数 =====

def initialize_environment(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    初始化环境
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    # 加载配置
    config = load_config(config_path)
    
    # 设置日志
    setup_logging(config)
    
    # 确保输出目录存在
    output_dir = config.get('output_dir', 'output')
    ensure_directory(output_dir)
    
    temp_dir = config.get('temp_dir', 'temp')
    ensure_directory(temp_dir)
    
    return config