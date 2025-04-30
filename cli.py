"""
命令行接口模块
提供交互式命令行界面
"""

import os
import sys
import logging
import argparse
import time
from typing import Dict, Any, List, Optional

# 导入自定义模块
from utils import load_config, is_video_file
# import main  <- Remove this line
from main import process_video as main_process_video, batch_process_videos as main_batch_process_videos # <- Add this line

logger = logging.getLogger(__name__)

class ProgressBar:
    """简单的命令行进度条"""
    
    def __init__(self, total: int, prefix: str = "", suffix: str = "", length: int = 50, fill: str = "█"):
        """
        初始化进度条
        
        Args:
            total: 总步骤数
            prefix: 前缀文字
            suffix: 后缀文字
            length: 进度条长度
            fill: 填充字符
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.start_time = time.time()
        self.iteration = 0
    
    def update(self, iteration: Optional[int] = None, prefix: Optional[str] = None, suffix: Optional[str] = None):
        """
        更新进度条
        
        Args:
            iteration: 当前步骤
            prefix: 更新前缀文字
            suffix: 更新后缀文字
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1
            
        if prefix is not None:
            self.prefix = prefix
            
        if suffix is not None:
            self.suffix = suffix
            
        percent = self.iteration / self.total
        filled_length = int(self.length * percent)
        bar = self.fill * filled_length + "-" * (self.length - filled_length)
        
        elapsed_time = time.time() - self.start_time
        if percent > 0:
            estimated_total = elapsed_time / percent
            remaining = estimated_total - elapsed_time
            time_str = f"{elapsed_time:.1f}s/{remaining:.1f}s"
        else:
            time_str = f"{elapsed_time:.1f}s"
        
        sys.stdout.write(f"\r{self.prefix} |{bar}| {percent:.1%} {self.suffix} {time_str}")
        sys.stdout.flush()
        
        if self.iteration == self.total:
            sys.stdout.write("\n")
    
    def finish(self, suffix: Optional[str] = None):
        """
        完成进度条
        
        Args:
            suffix: 完成时显示的后缀
        """
        final_suffix = suffix if suffix is not None else self.suffix
        self.update(self.total, suffix=final_suffix)

class CLI:
    """命令行界面类"""
    
    def __init__(self):
        """初始化命令行界面"""
        self.config = None
        self.config_path = "config.yaml"
    
    def setup_parser(self) -> argparse.ArgumentParser:
        """
        设置命令行参数解析器
        
        Returns:
            参数解析器
        """
        parser = argparse.ArgumentParser(
            description="MetaNote - 课程视频自动笔记生成系统",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # 命令参数
        subparsers = parser.add_subparsers(dest="command", help="子命令")
        
        # 处理单个视频命令
        process_parser = subparsers.add_parser("process", help="处理单个视频")
        process_parser.add_argument("video", help="视频文件路径")
        process_parser.add_argument("--output", "-o", help="输出目录")
        process_parser.add_argument("--no-progress", action="store_true", help="不显示进度信息")
        
        # 批处理命令
        batch_parser = subparsers.add_parser("batch", help="批量处理视频")
        batch_parser.add_argument("directory", help="视频文件目录")
        batch_parser.add_argument("--recursive", "-r", action="store_true", help="递归处理子目录")
        batch_parser.add_argument("--output", "-o", help="输出目录")
        
        # 启动ASR服务命令
        asr_parser = subparsers.add_parser("asr-server", help="启动ASR服务")
        asr_parser.add_argument("--model", help="ASR模型路径 (可选, 默认从配置文件读取)")
        asr_parser.add_argument("--device", default="cuda:0", help="运行设备")
        asr_parser.add_argument("--host", default="0.0.0.0", help="主机地址")
        asr_parser.add_argument("--port", type=int, default=8000, help="端口号")
        
        # 配置管理命令
        config_parser = subparsers.add_parser("config", help="配置管理")
        config_parser.add_argument("--show", action="store_true", help="显示当前配置")
        config_parser.add_argument("--init", action="store_true", help="初始化默认配置文件")
        config_parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="设置配置项")
        
        # 通用参数
        parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
        parser.add_argument("--asr-url", help="ASR服务器URL")
        parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="日志级别")
        parser.add_argument("--version", "-v", action="version", version="MetaNote v1.0.0")
        
        return parser
    
    def progress_callback(self, message: str, progress_bar: ProgressBar):
        """
        进度回调函数
        
        Args:
            message: 进度消息
            progress_bar: 进度条对象
        """
        # 根据消息更新进度
        if "步骤1" in message:
            progress_bar.update(1, suffix="提取音频...")
        elif "步骤2" in message:
            progress_bar.update(2, suffix="语音识别...")
        elif "步骤3" in message:
            progress_bar.update(3, suffix="提取关键帧...")
        elif "步骤4" in message:
            progress_bar.update(4, suffix="图像理解...")
        elif "步骤5" in message:
            progress_bar.update(5, suffix="生成笔记...")
        elif "处理完成" in message:
            progress_bar.finish("完成!")
    
    def process_video(self, args):
        """
        处理单个视频命令
        
        Args:
            args: 命令行参数
        """
        video_path = args.video
        output_dir = args.output or "output"
        
        # 检查视频文件
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在: {video_path}")
            return 1
        
        if not is_video_file(video_path):
            print(f"错误: 不支持的视频格式: {video_path}")
            return 1
        
        # 显示进度条
        if not args.no_progress:
            progress_bar = ProgressBar(5, prefix="处理视频:", suffix="正在开始...")
            progress_bar.update(0)
        
        try:
            # 处理视频
            # result = main.process_video(  <- Change this line
            result = main_process_video( # <- To this line
                video_path, 
                output_dir, 
                args.config,
                args.asr_url
            )
            
            if not args.no_progress:
                progress_bar.finish("完成!")
            
            if result["status"] == "success":
                print(f"\n✓ 处理成功！")
                print(f"笔记已保存到: {result['notes_path']}")
                print(f"处理耗时: {result['elapsed_time']:.2f}秒")
            else:
                print(f"\n✗ 处理失败: {result.get('error', '未知错误')}")
                return 1
            
        except Exception as e:
            if not args.no_progress:
                progress_bar.finish("失败!")
            print(f"\n✗ 处理视频时出错: {str(e)}")
            logger.exception("处理视频时出错")
            return 1
        
        return 0
    
    def batch_process(self, args):
        """
        批量处理视频命令
        
        Args:
            args: 命令行参数
        """
        directory = args.directory
        output_dir = args.output or "output"
        recursive = args.recursive
        
        try:
            # 批量处理视频
            # result = main.batch_process_videos( <- Change this line
            result = main_batch_process_videos( # <- To this line
                directory, 
                output_dir, 
                args.config,
                recursive,
                args.asr_url
            )
            
            print(f"\n批处理完成！")
            print(f"找到视频文件: {result['videos_found']}")
            print(f"成功处理: {result['processed']}")
            print(f"处理失败: {result['failed']}")
            
            # 如果所有视频都处理失败，返回错误码
            if result['videos_found'] > 0 and result['processed'] == 0:
                return 1
            
        except Exception as e:
            print(f"\n✗ 批量处理时出错: {str(e)}")
            logger.exception("批量处理时出错")
            return 1
        
        return 0
    
    def start_asr_server(self, args):
        """
        启动ASR服务命令
        
        Args:
            args: 命令行参数
        """
        try:
            # 导入ASR服务模块
            from asr_server import start_server
            from utils import get_config_value # <- Add this line

            model_path = args.model
            device = args.device
            host = args.host
            port = args.port

            # 如果命令行未提供模型路径，则从配置文件读取
            if not model_path:
                try:
                    config = load_config(args.config)
                    model_path = get_config_value(config, "asr.model_path")
                    # 如果配置文件中也没有，则报错
                    if not model_path:
                        print(f"错误: 未在命令行指定 --model 参数，且配置文件 {args.config} 中未找到 asr.model_path")
                        return 1
                    print(f"从配置文件 {args.config} 加载 ASR 模型路径: {model_path}")
                except FileNotFoundError:
                    print(f"错误: 配置文件 {args.config} 未找到，且未指定 --model 参数")
                    return 1
                except Exception as e:
                    print(f"错误: 加载配置文件 {args.config} 失败: {str(e)}")
                    logger.exception("加载配置文件失败")
                    return 1

            # 启动服务
            print(f"启动ASR服务: {host}:{port}")
            print(f"模型路径: {model_path}") # <- Use the determined model_path
            print(f"设备: {device}") # <- Use the determined device
            print("按Ctrl+C停止服务\n")

            start_server(model_path, device, host, port) # <- Use the determined variables

        except ImportError:
            print("错误: 无法导入ASR服务模块")
            return 1
        except Exception as e:
            print(f"错误: 启动ASR服务失败: {str(e)}")
            logger.exception("启动ASR服务失败")
            return 1

        return 0
    
    def manage_config(self, args):
        """
        配置管理命令
        
        Args:
            args: 命令行参数
        """
        from utils import save_config
        
        try:
            # 加载配置
            config = load_config(args.config)
            
            # 显示配置
            if args.show:
                import yaml
                print("当前配置:")
                print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
            
            # 初始化配置
            elif args.init:
                import shutil
                
                # 检查是否覆盖现有文件
                if os.path.exists(args.config):
                    response = input(f"配置文件 {args.config} 已存在，是否覆盖? (y/n): ")
                    if response.lower() != 'y':
                        print("已取消初始化配置")
                        return 0
                
                # 复制默认配置
                default_config = os.path.join(os.path.dirname(__file__), "config.yaml")
                if not os.path.exists(default_config):
                    print(f"错误: 默认配置文件不存在: {default_config}")
                    with open(args.config, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                else:
                    shutil.copy2(default_config, args.config)
                
                print(f"配置文件已初始化: {args.config}")
            
            # 设置配置项
            elif args.set:
                key, value = args.set
                
                # 尝试转换值的类型
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif '.' in value and all(p.isdigit() for p in value.split('.', 1)):
                    value = float(value)
                
                # 处理嵌套键
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                # 设置值
                current[keys[-1]] = value
                
                # 保存配置
                save_config(config, args.config)
                print(f"配置已更新: {key} = {value}")
                print(f"配置文件已保存: {args.config}")
            
            else:
                print("错误: 请指定配置操作 (--show, --init, --set)")
                return 1
            
        except Exception as e:
            print(f"错误: 配置管理失败: {str(e)}")
            logger.exception("配置管理失败")
            return 1
        
        return 0
    
    def run(self, args=None):
        """
        运行命令行界面
        
        Args:
            args: 命令行参数，如果为None则解析sys.argv
            
        Returns:
            返回码
        """
        parser = self.setup_parser()
        args = parser.parse_args(args)
        
        # 默认命令
        if not args.command:
            parser.print_help()
            return 1
        
        # 设置配置文件路径
        self.config_path = args.config
        
        # 配置日志
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        
        try:
            # 执行命令
            if args.command == "process":
                return self.process_video(args)
            elif args.command == "batch":
                return self.batch_process(args)
            elif args.command == "asr-server":
                # 传递 config 路径给 asr-server 命令处理函数
                return self.start_asr_server(args) # <- Pass args directly
            elif args.command == "config":
                return self.manage_config(args)
            else:
                print(f"错误: 未知命令: {args.command}")
                return 1
                
        except KeyboardInterrupt:
            print("\n用户中断，正在退出...")
            return 130  # 标准的中断返回码
        
        return 0

def main():
    """主入口函数"""
    cli = CLI()
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())