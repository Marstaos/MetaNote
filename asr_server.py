"""
ASR服务器模块
基于FastAPI的语音识别服务器，封装FunASR模型
"""

import os
import tempfile
import logging
import json
import time
import traceback
import sys
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger(__name__)

# 全局变量
funasr_available = False
AutoModel = None

# 首先尝试导入FunASR
try:
    from funasr import AutoModel
    funasr_available = True
    logger.info("成功导入FunASR库")
except ImportError as e:
    funasr_available = False
    logger.error(f"导入FunASR失败: {str(e)}")
    logger.error(f"导入路径: {sys.path}")
except Exception as e:
    funasr_available = False
    logger.error(f"导入FunASR时发生未知错误: {str(e)}")
    logger.error(traceback.format_exc())

class ASRServer:
    """ASR服务器类"""
    
    def __init__(self, model_path: str, device: str = "cuda:0", host: str = "0.0.0.0", port: int = 8000):
        """
        初始化ASR服务器
        
        Args:
            model_path: FunASR模型路径
            device: 运行设备（CPU或CUDA）
            host: 主机地址
            port: 端口号
        """
        if not funasr_available:
            raise ImportError("FunASR未安装或无法导入，请检查安装: pip install funasr")
            
        self.model_path = model_path
        self.device = device
        self.host = host
        self.port = port
        self.model = None
        self.app = FastAPI(title="MetaNote ASR API", description="语音识别服务")
        self.setup_app()
    
    def setup_app(self):
        """配置FastAPI应用"""
        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 注册启动事件
        @self.app.on_event("startup")
        async def startup_event():
            try:
                logger.info(f"正在加载ASR模型: {self.model_path}")
                self.model = AutoModel(model=self.model_path, device=self.device)
                logger.info("ASR模型加载成功")
            except Exception as e:
                logger.error(f"ASR模型加载失败: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
        # 注册健康检查接口
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "model_loaded": self.model is not None}
        
        # 注册语音识别接口
        @self.app.post("/asr/recognize")
        async def recognize_speech(file: UploadFile = File(...)) -> Dict[str, Any]:
            if not self.model:
                raise HTTPException(status_code=500, detail="ASR模型未加载")
            
            # 检查文件类型
            allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg']
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(status_code=400, detail=f"不支持的文件格式。支持的格式: {', '.join(allowed_extensions)}")
            
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                try:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file.flush()
                    
                    # 执行语音识别
                    logger.info(f"开始识别文件: {file.filename}")
                    start_time = time.time()
                    try:
                        result = self.model.generate(input=temp_file.name, batch_size_s=300)
                    except Exception as e:
                        logger.error(f"ASR模型处理失败: {str(e)}")
                        logger.error(traceback.format_exc())
                        raise HTTPException(status_code=500, detail=f"ASR处理错误: {str(e)}")
                        
                    end_time = time.time()
                    
                    logger.info(f"识别完成: {file.filename}, 耗时: {end_time - start_time:.2f}秒")
                    
                    return {
                        "status": "success",
                        "filename": file.filename,
                        "processing_time": f"{end_time - start_time:.2f}秒",
                        "result": result
                    }
                    
                except Exception as e:
                    logger.error(f"识别失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")
                
                finally:
                    # 清理临时文件
                    try:
                        os.unlink(temp_file.name)
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {str(e)}")
    
    def run(self):
        """启动ASR服务器"""
        logger.info(f"启动ASR服务器: http://{self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


def start_server(model_path: str, device: str = "cuda:0", host: str = "0.0.0.0", port: int = 8000):
    """
    启动ASR服务
    
    Args:
        model_path: 模型路径
        device: 运行设备
        host: 服务器地址
        port: 端口号
    """
    # 更详细的导入诊断
    if not funasr_available:
        logger.error("FunASR未安装或无法导入")
        logger.error("请确认已安装FunASR: pip install funasr")
        logger.error("Python路径: " + str(sys.path))
        logger.error("Python版本: " + sys.version)
        
        # 尝试直接导入以获取更详细错误
        try:
            import funasr
            logger.info(f"FunASR已安装，版本: {funasr.__version__ if hasattr(funasr, '__version__') else '未知'}")
            logger.info(f"但无法导入AutoModel组件")
        except ImportError as e:
            logger.error(f"导入funasr包失败: {str(e)}")
        except Exception as e:
            logger.error(f"导入funasr时出现其他错误: {str(e)}")
            
        print("FunASR导入失败，请检查安装和环境。您可以尝试以下操作：")
        print("1. 验证安装: python -c 'import funasr; print(funasr.__file__)'")
        print("2. 重新安装: pip uninstall funasr && pip install funasr")
        print("3. 检查环境变量，确保使用了正确的Python环境")
        return
        
    try:
        server = ASRServer(model_path, device, host, port)
        server.run()
    except Exception as e:
        logger.error(f"启动ASR服务失败: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    import argparse
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="启动ASR服务")
    parser.add_argument("--model", required=True, help="ASR模型路径")
    parser.add_argument("--device", default="cuda:0", help="运行设备")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    
    args = parser.parse_args()
    
    # 启动服务
    start_server(args.model, args.device, args.host, args.port)