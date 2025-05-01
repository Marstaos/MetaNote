"""
MetaNote Web Frontend
A modern web interface for the MetaNote application using Streamlit
"""

import os
import sys
import time
import yaml
import json
import subprocess
import threading
import streamlit as st
from pathlib import Path
import re
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse

# Import custom modules from MetaNote
from utils import load_config, save_config, get_config_value, set_config_value, is_video_file
from asr_client import ASRClient
from main import process_video, batch_process_videos

# Page Configuration
st.set_page_config(
    page_title="MetaNote - 课程视频笔记生成器",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* 保持原有样式不变 */
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'asr' not in st.session_state:
    st.session_state.asr = {
        'process': None,
        'status': "stopped",  # running/stopped/starting/error
        'url': None,
        'monitor_thread': None,
        'log_file': None
    }

# Helper functions

def find_asr_model_paths() -> List[str]:
    """自动查找可能的ASR模型路径"""
    search_paths = [
        # 常见模型存储路径
        "/usr/share/models",
        "/opt/models",
        str(Path.home() / "models"),
        "models",
        "asr_models",
        # 项目内可能路径
        "pretrained_models",
        "FunASR/resources/models",
        # 用户自定义路径
        str(Path(st.session_state.config.get("asr", {}).get("model_path", "")).parent)
    ]
    
    found_paths = []
    for path in search_paths:
        full_path = Path(path).absolute()
        # 检查是否为有效的FunASR模型目录
        if full_path.exists() and any((full_path / "config.yaml").exists() for p in full_path.glob("*")):
            found_paths.append(str(full_path))
    
    # 去重并保留唯一路径
    return list({p: True for p in found_paths if p}.keys())

def extract_host_port_from_url(url: str) -> Tuple[str, int]:
    """从URL中提取主机和端口"""
    parsed_url = urlparse(url)
    host = parsed_url.hostname or "localhost"
    port = parsed_url.port or 8000  # 默认端口设为8000
    return host, port

def normalize_server_url(url: str) -> str:
    """规范化服务器URL（0.0.0.0 -> localhost）"""
    parsed = urlparse(url)
    if parsed.hostname == "0.0.0.0":
        return f"http://localhost:{parsed.port}"
    return url

def enhanced_health_check(server_url: str) -> bool:
    """增强型健康检查，带指数退避重试"""
    from math import pow
    max_retries = 5
    server_url = normalize_server_url(server_url)
    
    for i in range(max_retries):
        try:
            client = ASRClient(server_url)
            if client.check_health():
                return True
            time.sleep(pow(2, i))  # 指数退避
        except Exception:
            if i == max_retries - 1:
                return False
    return False

def start_status_monitor():
    """启动后台状态监控线程"""
    def monitor_task():
        while True:
            # 检查会话状态是否存在asr属性
            if not hasattr(st.session_state, 'asr'):
                break
                
            current_status = st.session_state.asr.get('status', 'stopped')
            # 仅在运行状态下保持监控
            if current_status not in ['running', 'starting']:
                break

            try:
                current_url = st.session_state.asr.get('url')
                if not current_url:
                    break
                
                client = ASRClient(normalize_server_url(current_url))
                new_status = "running" if client.check_health() else "error"
                
                if new_status != current_status:
                    st.session_state.asr['status'] = new_status
                    st.rerun()
                
            except Exception as e:
                print(f"监控线程异常: {str(e)}")
                st.session_state.asr['status'] = "error"
                st.rerun()
            
            time.sleep(3)
    
    # 确保只有一个监控线程运行
    if not st.session_state.asr.get('monitor_thread') or not st.session_state.asr['monitor_thread'].is_alive():
        monitor_thread = threading.Thread(target=monitor_task, daemon=True)
        monitor_thread.start()
        st.session_state.asr['monitor_thread'] = monitor_thread

def start_asr_server_subprocess(model_path: str, device: str = "cuda:0", host: str = "0.0.0.0", port: int = 8000):
    """启动ASR服务子进程"""
    st.session_state.asr.update({
        'status': "starting",
        'url': f"http://{host}:{port}",
        'log_file': None
    })
    
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"asr_server_{int(time.time())}.log")
    st.session_state.asr['log_file'] = log_file
    
    # 构建命令
    cmd = [
        sys.executable,
        "asr_server.py",
        "--model", model_path,
        "--device", device,
        "--host", host,
        "--port", str(port)
    ]
    
    # 启动进程
    try:
        with open(log_file, 'w') as log_fd:
            process = subprocess.Popen(
                cmd,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            st.session_state.asr['process'] = process
            
        # 等待初始化
        time.sleep(8)
        
        # 健康检查
        if enhanced_health_check(st.session_state.asr['url']):
            st.session_state.asr['status'] = "running"
            start_status_monitor()
        else:
            st.session_state.asr['status'] = "error"
            
    except Exception as e:
        st.error(f"启动失败: {str(e)}")
        st.session_state.asr['status'] = "error"

def stop_asr_server():
    """停止ASR服务"""
    process = st.session_state.asr['process']
    if process:
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception as e:
            st.error(f"停止服务时出错: {str(e)}")
    
    st.session_state.asr.update({
        'process': None,
        'status': "stopped",
        'url': None,
        'monitor_thread': None
    })

def run_process_with_progress(
    video_path: str, 
    output_dir: str, 
    config_path: str,
    asr_url: str,
    progress_bar: st.progress,
    status_text: st.empty
) -> Dict[str, Any]:
    """Run the video processing with progress updates"""
    # Define the steps and their approximate weight in the process
    steps = {
        "准备": 5,
        "提取音频": 10,
        "语音识别": 30,
        "提取关键帧": 25,
        "图像理解": 20,
        "生成笔记": 10
    }
    
    current_step = "准备"
    progress_bar.progress(0)
    status_text.text(f"正在{current_step}...")
    
    # Process the video
    result = process_video(video_path, output_dir, config_path, asr_url)
    
    # Update progress based on result
    if result["status"] == "success":
        progress_bar.progress(100)
        status_text.text("处理完成！")
    else:
        status_text.text(f"处理失败：{result.get('error', '未知错误')}")
    
    return result

def read_last_lines(file_path, n=20):
    """Read the last n lines from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-n:]
    except Exception as e:
        return [f"Error reading log file: {str(e)}"]

def check_asr_health(server_url: str) -> bool:
    """Check ASR server health and update status"""
    global ASR_SERVER_STATUS
    
    try:
        client = ASRClient(server_url)
        health = client.check_health()
        
        if health:
            ASR_SERVER_STATUS = "running"
            return True
        else:
            if ASR_SERVER_STATUS == "starting":
                # Keep status as starting if it's in starting state
                pass
            else:
                ASR_SERVER_STATUS = "error"
            return False
    except Exception as e:
        if ASR_SERVER_STATUS == "starting":
            # Keep status as starting if it's in starting state
            pass
        else:
            ASR_SERVER_STATUS = "error"
        return False

# Session state initialization
if 'config' not in st.session_state:
    # Load config file
    config_path = "config.yaml"
    st.session_state.config = load_config(config_path)
    st.session_state.config_path = config_path

if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []

if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "idle"

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

if 'asr_log_file' not in st.session_state:
    st.session_state.asr_log_file = None

if 'asr_server_cmd' not in st.session_state:
    st.session_state.asr_server_cmd = None

if 'asr_server_url' not in st.session_state:
    st.session_state.asr_server_url = None

# UI Components
def render_header():
    """Render the application header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown('<div class="logo-text">MetaNote</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">课程视频自动笔记生成工具</div>', unsafe_allow_html=True)

def render_config_section():
    """Render the configuration section"""
    st.header("配置设置")
    
    config = st.session_state.config
    config_path = st.session_state.config_path
    
    # File path selection
    new_config_path = st.text_input("配置文件路径", value=config_path)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("加载配置", use_container_width=True):
            st.session_state.config = load_config(new_config_path)
            st.session_state.config_path = new_config_path
            st.rerun()
    
    with col2:
        if st.button("保存配置", use_container_width=True):
            success = save_config(config, new_config_path)
            if success:
                st.toast("配置保存成功！", icon="✅")
            else:
                st.toast("配置保存失败！", icon="❌")
    
    # Config sections
    with st.expander("输出设置", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            output_dir = st.text_input(
                "输出目录", 
                value=config.get("output_dir", "output")
            )
            config["output_dir"] = output_dir
        
        with col2:
            temp_dir = st.text_input(
                "临时文件目录", 
                value=config.get("temp_dir", "temp")
            )
            config["temp_dir"] = temp_dir
    
    with st.expander("ASR 设置", expanded=True):
        # Ensure the asr section exists
        if "asr" not in config:
            config["asr"] = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            server_url = st.text_input(
                "ASR 服务器 URL", 
                value=config.get("asr", {}).get("server_url", "http://localhost:8000")
            )
            config["asr"]["server_url"] = server_url
            
            # Extract host and port from URL
            host, port = extract_host_port_from_url(server_url)
            st.text_input("服务器端口", value=str(port), disabled=True,
                         help="端口已从服务器 URL 提取。要更改端口，请修改上面的 URL。")
        
        with col2:
            # Try to find potential model paths
            model_paths = find_asr_model_paths()
            default_path = config.get("asr", {}).get("model_path", "")
            
            if model_paths and not default_path:
                default_path = model_paths[0]
            
            model_path = st.text_input(
                "ASR 模型路径", 
                value=default_path,
                help="FunASR 模型路径，例如 SenseVoiceSmall 目录"
            )
            config["asr"]["model_path"] = model_path
            
            if model_paths:
                selected_path = st.selectbox(
                    "检测到的模型路径", 
                    options=["自定义"] + model_paths,
                    index=0
                )
                if selected_path != "自定义":
                    config["asr"]["model_path"] = selected_path
        
        col1, col2 = st.columns(2)
        
        with col1:
            device = st.text_input(
                "运行设备", 
                value=config.get("asr", {}).get("device", "cuda:0"),
                help="设置为 cpu 如果没有 CUDA 支持"
            )
            config["asr"]["device"] = device
    
    with st.expander("视频处理设置"):
        if "video" not in config:
            config["video"] = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_rate = st.number_input(
                "每秒采样的帧数", 
                min_value=1, 
                max_value=10, 
                value=config.get("video", {}).get("sample_rate", 1)
            )
            config["video"]["sample_rate"] = int(sample_rate)
            
            stable_duration = st.number_input(
                "内容稳定的最小持续秒数", 
                min_value=1, 
                max_value=10, 
                value=config.get("video", {}).get("stable_duration", 3)
            )
            config["video"]["stable_duration"] = int(stable_duration)
        
        with col2:
            scene_threshold = st.slider(
                "场景切换检测的阈值", 
                min_value=0.1, 
                max_value=0.9, 
                value=float(config.get("video", {}).get("scene_threshold", 0.3)),
                step=0.05
            )
            config["video"]["scene_threshold"] = float(scene_threshold)
            
            parallel_requests = st.number_input(
                "并行处理线程数", 
                min_value=1, 
                max_value=8, 
                value=config.get("video", {}).get("parallel_requests", 2)
            )
            config["video"]["parallel_requests"] = int(parallel_requests)
    
    with st.expander("图像理解设置"):
        if "image_understanding" not in config:
            config["image_understanding"] = {}
        
        provider_options = ["qwen", "ollama"]
        current_provider = config.get("image_understanding", {}).get("provider", "qwen")
        
        provider = st.selectbox(
            "图像理解服务", 
            options=provider_options,
            index=provider_options.index(current_provider) if current_provider in provider_options else 0
        )
        config["image_understanding"]["provider"] = provider
        
        if provider == "qwen":
            if "qwen" not in config.get("image_understanding", {}):
                config["image_understanding"]["qwen"] = {}
            
            col1, col2 = st.columns(2)
            
            with col1:
                api_key = st.text_input(
                    "千问 API 密钥", 
                    value=config.get("image_understanding", {}).get("qwen", {}).get("api_key", ""),
                    type="password"
                )
                config["image_understanding"]["qwen"]["api_key"] = api_key
            
            with col2:
                base_url = st.text_input(
                    "API Base URL", 
                    value=config.get("image_understanding", {}).get("qwen", {}).get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                )
                config["image_understanding"]["qwen"]["base_url"] = base_url
            
            model = st.text_input(
                "模型名称", 
                value=config.get("image_understanding", {}).get("qwen", {}).get("model", "qwen-vl-plus")
            )
            config["image_understanding"]["qwen"]["model"] = model
        
        elif provider == "ollama":
            if "ollama" not in config.get("image_understanding", {}):
                config["image_understanding"]["ollama"] = {}
            
            col1, col2 = st.columns(2)
            
            with col1:
                url = st.text_input(
                    "Ollama API URL", 
                    value=config.get("image_understanding", {}).get("ollama", {}).get("url", "http://localhost:11434/api/generate")
                )
                config["image_understanding"]["ollama"]["url"] = url
            
            with col2:
                model = st.text_input(
                    "模型名称", 
                    value=config.get("image_understanding", {}).get("ollama", {}).get("model", "llava:13b")
                )
                config["image_understanding"]["ollama"]["model"] = model
    
    with st.expander("笔记生成设置"):
        if "note_generator" not in config:
            config["note_generator"] = {}
        
        provider_options = ["qwen"]
        current_provider = config.get("note_generator", {}).get("provider", "qwen")
        
        provider = st.selectbox(
            "笔记生成服务", 
            options=provider_options,
            index=provider_options.index(current_provider) if current_provider in provider_options else 0
        )
        config["note_generator"]["provider"] = provider
        
        if provider == "qwen":
            if "qwen" not in config.get("note_generator", {}):
                config["note_generator"]["qwen"] = {}
            
            col1, col2 = st.columns(2)
            
            with col1:
                api_key = st.text_input(
                    "千问 API 密钥 (笔记生成)", 
                    value=config.get("note_generator", {}).get("qwen", {}).get("api_key", ""),
                    type="password"
                )
                config["note_generator"]["qwen"]["api_key"] = api_key
            
            with col2:
                base_url = st.text_input(
                    "API Base URL (笔记生成)", 
                    value=config.get("note_generator", {}).get("qwen", {}).get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                )
                config["note_generator"]["qwen"]["base_url"] = base_url
            
            model = st.text_input(
                "模型名称 (笔记生成)", 
                value=config.get("note_generator", {}).get("qwen", {}).get("model", "qwen-plus")
            )
            config["note_generator"]["qwen"]["model"] = model
            
            system_prompt = st.text_area(
                "系统提示语", 
                value=config.get("note_generator", {}).get("qwen", {}).get("system_prompt", "你是一位专业的教育内容编辑，擅长将课程视频的文字记录和图像整理成结构化的学习笔记。"),
                height=150
            )
            config["note_generator"]["qwen"]["system_prompt"] = system_prompt
    
    with st.expander("日志设置"):
        if "logging" not in config:
            config["logging"] = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            level_options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            current_level = config.get("logging", {}).get("level", "INFO")
            
            level = st.selectbox(
                "日志级别", 
                options=level_options,
                index=level_options.index(current_level) if current_level in level_options else 1
            )
            config["logging"]["level"] = level
        
        with col2:
            log_file = st.text_input(
                "日志文件", 
                value=config.get("logging", {}).get("file", "metanote.log")
            )
            config["logging"]["file"] = log_file
    
    st.session_state.config = config

def render_asr_server_section():
    """ASR服务管理界面"""
    st.header("ASR 服务管理")
    
    status_map = {
        "running": ("运行中", "status-running"),
        "stopped": ("已停止", "status-stopped"),
        "starting": ("启动中...", "status-running"),
        "error": ("启动失败", "status-stopped")
    }
    
    status_text, status_class = status_map.get(
        st.session_state.asr['status'],
        ("未知状态", "status-stopped")
    )
    
    # 状态显示
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"​**服务状态:​**​ <span class='{status_class}'>{status_text}</span>", unsafe_allow_html=True)
        if st.session_state.asr['url']:
            st.markdown(f"​**服务地址:​**​ `{normalize_server_url(st.session_state.asr['url'])}`")
    
    with col2:
        if st.session_state.asr['status'] in ["stopped", "error"]:
            if st.button("🚀 启动服务", use_container_width=True):
                config = st.session_state.config
                model_path = config.get("asr", {}).get("model_path", "")
                if not model_path:
                    st.error("请先配置模型路径")
                else:
                    server_url = config.get("asr", {}).get("server_url", "http://localhost:8000")
                    host, port = extract_host_port_from_url(server_url)
                    start_asr_server_subprocess(model_path, 
                                               config.get("asr", {}).get("device", "cuda:0"),
                                               host, port)
        else:
            if st.button("🛑 停止服务", use_container_width=True):
                stop_asr_server()
    
    # 日志显示
    if st.session_state.asr['log_file']:
        with st.expander("服务日志", expanded=True):
            if st.button("刷新日志"):
                pass
            
            try:
                with open(st.session_state.asr['log_file'], 'r') as f:
                    lines = f.readlines()[-20:]
                    st.code("".join(lines), language="log")
            except Exception as e:
                st.error(f"读取日志失败: {str(e)}")

# 修改处理视频部分的检查逻辑
def render_process_section():
    st.header("处理视频")
    
    is_asr_ready = st.session_state.asr['status'] == "running"
    
    if not is_asr_ready:
        st.warning("ASR服务未就绪，请先启动服务")
    else:
        st.success(f"ASR服务已连接: {normalize_server_url(st.session_state.asr['url'])}")
    
    # Display the current ASR URL that will be used
    config = st.session_state.config
    asr_url = st.session_state.asr.get('url') or config.get("asr", {}).get("server_url", "http://localhost:8000")
    st.info(f"将使用 ASR 服务: {asr_url}")
    
    # Tabs for single/batch processing
    tab1, tab2 = st.tabs(["单个视频处理", "批量处理"])
    
    with tab1:
        st.subheader("处理单个视频")
        
        uploaded_file = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov", "mkv", "webm"])
        video_path = st.text_input("或输入视频文件路径", placeholder="/path/to/video.mp4")
        
        col1, col2 = st.columns(2)
        
        with col1:
            output_dir = st.text_input(
                "输出目录", 
                value=st.session_state.config.get("output_dir", "output"),
                help="生成的笔记将保存在这个目录中"
            )
        
        # Process button
        if st.button("开始处理", disabled=not is_asr_ready or (not uploaded_file and not video_path), use_container_width=True):
            # Get configuration
            config = st.session_state.config
            config_path = st.session_state.config_path
            
            # Determine video path
            process_path = None
            if uploaded_file:
                # Save uploaded file to temporary location
                temp_dir = config.get("temp_dir", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                process_path = temp_path
            elif video_path and os.path.exists(video_path):
                process_path = video_path
            
            if process_path and is_video_file(process_path):
                # Setup progress display
                st.session_state.processing_status = "processing"
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process the video
                with st.spinner("正在处理视频..."):
                    result = run_process_with_progress(
                        process_path,
                        output_dir,
                        config_path,
                        asr_url,
                        progress_bar,
                        status_text
                    )
                
                # Display results
                if result["status"] == "success":
                    st.session_state.processing_results.append(result)
                    st.success(f"✅ 处理成功！笔记已保存到: {result['notes_path']}")
                    
                    # Display the notes
                    with st.expander("查看生成的笔记", expanded=True):
                        try:
                            with open(result['notes_path'], 'r', encoding='utf-8') as f:
                                notes_content = f.read()
                            st.markdown(notes_content)
                        except Exception as e:
                            st.error(f"无法加载笔记: {str(e)}")
                    
                    # Display processing information
                    with st.expander("处理详情"):
                        st.json(result)
                else:
                    st.error(f"❌ 处理失败: {result.get('error', '未知错误')}")
                
                st.session_state.processing_status = "idle"
            else:
                st.error("请选择有效的视频文件！")
    
    with tab2:
        st.subheader("批量处理视频")
        
        directory = st.text_input("视频目录路径", placeholder="/path/to/videos/folder")
        recursive = st.checkbox("递归处理子目录", value=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_output_dir = st.text_input(
                "批处理输出目录", 
                value=st.session_state.config.get("output_dir", "output"),
                help="生成的笔记将保存在这个目录中"
            )
        
        # Process button
        if st.button("开始批处理", disabled=not is_asr_ready or not directory, use_container_width=True):
            if os.path.isdir(directory):
                # Get configuration
                config = st.session_state.config
                config_path = st.session_state.config_path
                
                # Process videos
                with st.spinner("正在批量处理视频..."):
                    result = batch_process_videos(
                        directory,
                        batch_output_dir,
                        config_path,
                        recursive,
                        asr_url
                    )
                
                # Display results
                if result["videos_found"] > 0:
                    if result["processed"] > 0:
                        st.success(f"✅ 批处理完成！成功处理 {result['processed']} 个视频，失败 {result['failed']} 个。")
                    else:
                        st.error(f"❌ 批处理失败！所有 {result['videos_found']} 个视频处理失败。")
                    
                    # Display detailed results
                    with st.expander("查看详细结果"):
                        st.json(result)
                else:
                    st.warning(f"⚠️ 在目录中未找到视频文件: {directory}")
            else:
                st.error(f"❌ 目录不存在: {directory}")

def render_results_section():
    """Render the results section"""
    st.header("处理结果")
    
    results = st.session_state.processing_results
    
    if not results:
        st.info("暂无处理结果。请在\"处理视频\"选项卡中处理视频。")
        return
    
    # Display each result
    for i, result in enumerate(results):
        with st.expander(f"结果 #{i+1}: {os.path.basename(result['video_path'])}", expanded=(i == 0)):
            st.markdown(f"**视频:** {result['video_path']}")
            st.markdown(f"**状态:** {'✅ 成功' if result['status'] == 'success' else '❌ 失败'}")
            
            if result['status'] == 'success':
                st.markdown(f"**笔记路径:** {result['notes_path']}")
                st.markdown(f"**处理时间:** {result['elapsed_time']:.2f} 秒")
                
                # Button to open notes
                if st.button(f"查看笔记 #{i+1}", key=f"view_notes_{i}"):
                    try:
                        with open(result['notes_path'], 'r', encoding='utf-8') as f:
                            notes_content = f.read()
                        st.markdown(notes_content)
                    except Exception as e:
                        st.error(f"无法加载笔记: {str(e)}")
            else:
                st.markdown(f"**错误:** {result.get('error', '未知错误')}")

def main():
    """Main application function"""
    render_header()
    
    # Main navigation
    st.sidebar.title("导航")
    pages = ["配置设置", "ASR 服务管理", "处理视频", "处理结果"]
    selection = st.sidebar.radio("选择页面", pages, index=st.session_state.current_tab)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "📝 **MetaNote** 是一个能够从课程视频中自动生成带有关键图像的结构化 Markdown 笔记的工具。"
        "\n\n使用步骤："
        "\n1. 在配置设置中设置好参数"
        "\n2. 启动 ASR 服务"
        "\n3. 选择视频进行处理"
    )
    
    # Store current tab
    st.session_state.current_tab = pages.index(selection)
    
    # Render selected page
    if selection == "配置设置":
        render_config_section()
    elif selection == "ASR 服务管理":
        render_asr_server_section()
    elif selection == "处理视频":
        render_process_section()
    elif selection == "处理结果":
        render_results_section()

if __name__ == "__main__":
    main()