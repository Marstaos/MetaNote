"""
MetaNote主程序
课程视频自动笔记生成系统
"""

import os
import sys
import logging
import argparse
import time
from typing import Dict, Any, Optional
from pathlib import Path

# 导入自定义模块
from utils import load_config, setup_logging, extract_audio, is_video_file
from asr_client import ASRClient
from frame_extractor import FrameExtractor
from image_processor import create_image_processor
from note_generator import create_note_generator

logger = logging.getLogger(__name__)

def process_video(
    video_path: str, 
    output_dir: str = "output", 
    config_path: str = "config.yaml",
    asr_server_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    处理视频生成笔记的主函数
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        config_path: 配置文件路径
        asr_server_url: ASR服务器URL，可选
        
    Returns:
        处理结果
    """
    # 加载配置
    config = load_config(config_path)
    setup_logging(config)
    
    # 设置输出目录
    if output_dir:
        config['output_dir'] = output_dir
    
    # 检查视频文件
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    if not is_video_file(video_path):
        raise ValueError(f"不支持的视频格式: {video_path}")
    
    start_time = time.time()
    logger.info(f"开始处理视频: {video_path}")
    
    result = {
        "video_path": video_path,
        "output_dir": output_dir,
        "status": "processing",
        "start_time": start_time,
    }
    
    try:
        # 1. 初始化ASR客户端
        if asr_server_url:
            config['asr'] = config.get('asr', {})
            config['asr']['server_url'] = asr_server_url
        
        asr_url = config.get('asr', {}).get('server_url', 'http://localhost:8000')
        asr_client = ASRClient(asr_url)
        
        # 检查ASR服务
        if not asr_client.check_health():
            raise RuntimeError(f"ASR服务不可用: {asr_url}")
        
        # 2. 提取音频
        temp_dir = config.get('temp_dir', 'temp')
        audio_path = extract_audio(
            video_path, 
            os.path.join(temp_dir, f"{Path(video_path).stem}_audio.wav")
        )
        result["audio_path"] = audio_path
        
        # 3. 进行语音识别
        logger.info("正在进行语音识别...")
        transcript = asr_client.recognize_audio(audio_path)
        if not transcript:
            raise RuntimeError("语音识别失败")
        result["transcript"] = transcript
        
        # 4. 提取关键帧
        logger.info("正在提取关键帧...")
        video_config = config.get('video', {})
        frame_extractor = FrameExtractor(
            video_path,
            output_dir,
            sample_rate=video_config.get('sample_rate', 1),
            stable_duration=video_config.get('stable_duration', 3),
            scene_threshold=video_config.get('scene_threshold', 0.3)
        )
        frames = frame_extractor.extract_frames()
        result["frames"] = frames
        
        # 5. 图像理解与评估
        logger.info("正在进行图像理解...")
        image_processor = create_image_processor(config)
        image_processor.process_frame_extractor(frame_extractor)
        
        # 6. 生成笔记
        logger.info("正在生成笔记...")
        note_generator = create_note_generator(config)
        notes_path = note_generator.process_frame_extractor_and_transcript(
            frame_extractor, 
            transcript
        )
        result["notes_path"] = notes_path
        
        # 7. 完成处理
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        result["status"] = "success"
        result["elapsed_time"] = elapsed_time
        result["end_time"] = end_time
        
        logger.info(f"处理完成! 耗时: {elapsed_time:.2f}秒")
        logger.info(f"生成的笔记: {notes_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"处理视频时发生错误: {str(e)}")
        
        end_time = time.time()
        result["status"] = "error"
        result["error"] = str(e)
        result["end_time"] = end_time
        result["elapsed_time"] = end_time - start_time
        
        return result

def batch_process_videos(
    directory: str, 
    output_dir: str = "output", 
    config_path: str = "config.yaml",
    recursive: bool = False,
    asr_server_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    批量处理目录中的视频
    
    Args:
        directory: 视频目录
        output_dir: 输出目录
        config_path: 配置文件路径
        recursive: 是否递归处理子目录
        asr_server_url: ASR服务器URL，可选
        
    Returns:
        批处理结果
    """
    # 检查目录
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise NotADirectoryError(f"目录不存在: {directory}")
    
    # 查找视频文件
    video_files = []
    
    if recursive:
        # 递归查找
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if is_video_file(file_path):
                    video_files.append(file_path)
    else:
        # 只查找当前目录
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and is_video_file(file_path):
                video_files.append(file_path)
    
    if not video_files:
        logger.warning(f"目录中没有找到视频文件: {directory}")
        return {
            "status": "warning",
            "message": f"目录中没有找到视频文件: {directory}",
            "videos_found": 0,
            "processed": 0,
            "results": []
        }
    
    logger.info(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    results = []
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(video_files):
        logger.info(f"处理视频 {i+1}/{len(video_files)}: {video_path}")
        
        try:
            result = process_video(
                video_path, 
                output_dir, 
                config_path,
                asr_server_url
            )
            
            if result.get("status") == "success":
                successful += 1
            else:
                failed += 1
                
            results.append(result)
            
        except Exception as e:
            logger.error(f"处理视频 {video_path} 时出错: {str(e)}")
            failed += 1
            results.append({
                "video_path": video_path,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "status": "completed",
        "videos_found": len(video_files),
        "processed": successful,
        "failed": failed,
        "results": results
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MetaNote - 课程视频自动笔记生成")
    
    # 命令参数
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 处理单个视频命令
    process_parser = subparsers.add_parser("process", help="处理单个视频")
    process_parser.add_argument("video", help="视频文件路径")
    process_parser.add_argument("--output", "-o", help="输出目录")
    
    # 批处理命令
    batch_parser = subparsers.add_parser("batch", help="批量处理视频")
    batch_parser.add_argument("directory", help="视频文件目录")
    batch_parser.add_argument("--recursive", "-r", action="store_true", help="递归处理子目录")
    batch_parser.add_argument("--output", "-o", help="输出目录")
    
    # 启动ASR服务命令
    asr_parser = subparsers.add_parser("asr-server", help="启动ASR服务")
    asr_parser.add_argument("--model", required=True, help="ASR模型路径")
    asr_parser.add_argument("--device", default="cuda:0", help="运行设备")
    asr_parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    asr_parser.add_argument("--port", type=int, default=8000, help="端口号")
    
    # 通用参数
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    parser.add_argument("--asr-url", help="ASR服务器URL")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 默认命令
    if not args.command:
        parser.print_help()
        return
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # 执行命令
    try:
        if args.command == "process":
            # 处理单个视频
            result = process_video(
                args.video, 
                args.output or "output", 
                args.config,
                args.asr_url
            )
            
            if result["status"] == "success":
                print(f"\n✓ 处理成功！")
                print(f"笔记已保存到: {result['notes_path']}")
                print(f"处理耗时: {result['elapsed_time']:.2f}秒")
            else:
                print(f"\n✗ 处理失败: {result.get('error', '未知错误')}")
            
        elif args.command == "batch":
            # 批量处理视频
            result = batch_process_videos(
                args.directory, 
                args.output or "output", 
                args.config,
                args.recursive,
                args.asr_url
            )
            
            print(f"\n批处理完成！")
            print(f"找到视频文件: {result['videos_found']}")
            print(f"成功处理: {result['processed']}")
            print(f"处理失败: {result['failed']}")
            
        elif args.command == "asr-server":
            # 启动ASR服务
            from asr_server import start_server
            start_server(args.model, args.device, args.host, args.port)
            
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        logger.exception("执行命令时出错")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())