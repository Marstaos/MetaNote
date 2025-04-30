"""
帧提取模块
从视频中提取有意义的关键帧
"""

import cv2
import numpy as np
import os
import time
import logging
import json
import shutil
from datetime import timedelta
from typing import List, Dict, Any, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

logger = logging.getLogger(__name__)

class FrameExtractor:
    """从视频中提取有意义的关键帧"""
    
    def __init__(
        self, 
        video_path: str, 
        output_dir: str = "output", 
        sample_rate: int = 1, 
        stable_duration: int = 3, 
        scene_threshold: float = 0.3
    ):
        """
        初始化帧提取器
        
        Args:
            video_path: 输入视频的路径
            output_dir: 保存图像的主目录
            sample_rate: 每秒采样的帧数
            stable_duration: 定义内容稳定的最小持续秒数
            scene_threshold: 场景切换检测的阈值 (0-1)
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.stable_duration = stable_duration
        self.scene_threshold = scene_threshold
        
        # 提取视频文件名作为前缀
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 创建特定于此视频的输出目录结构
        self.video_output_dir = os.path.join(self.output_dir, self.video_name)
        self.frames_dir = os.path.join(self.video_output_dir, "frames")
        self.valuable_frames_dir = os.path.join(self.video_output_dir, "valuable_frames")
        
        # 创建目录结构
        for dir_path in [self.video_output_dir, self.frames_dir, self.valuable_frames_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 打开视频
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        
        logger.info(f"视频信息: {self.frame_count} 帧, {self.fps} FPS, 时长: {self.duration:.2f}秒")
        
        # 存储候选帧
        self.candidate_frames = []
        
        # 存储已保存的图像的相似度哈希，用于去重
        self.saved_hashes = set()
        
        # 存储元数据
        self.metadata = {
            "video_info": {
                "filename": self.video_path,
                "duration": self.duration,
                "fps": self.fps,
                "frame_count": self.frame_count
            },
            "extraction_params": {
                "sample_rate": self.sample_rate,
                "stable_duration": self.stable_duration,
                "scene_threshold": self.scene_threshold
            },
            "frames": []
        }
    
    def extract_frames(self) -> List[Dict[str, Any]]:
        """
        提取关键帧的主函数
        
        Returns:
            提取的关键帧列表
        """
        # 设置采样间隔
        sample_interval = int(self.fps / self.sample_rate)
        
        # 初始化
        frames_buffer = []  # 用于存储连续帧以分析稳定性
        prev_frame = None
        prev_hist = None
        stable_start = None
        
        # 候选帧和得分
        candidates = []
        
        frame_idx = 0
        
        logger.info("开始提取关键帧...")
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # 只处理采样间隔的帧
            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue
            
            current_time = frame_idx / self.fps
            timestamp = str(timedelta(seconds=int(current_time)))
            
            # 转换为灰度图，用于分析
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 场景切换检测
            scene_change = False
            if prev_frame is not None:
                # 计算直方图
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if prev_hist is not None:
                    # 比较直方图
                    hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    
                    # 如果相关性低于阈值，认为是场景切换
                    if hist_diff < (1 - self.scene_threshold):
                        scene_change = True
                        score = 20  # 场景切换基础分
                        
                        # 视觉复杂度分析
                        complexity_score = self._analyze_complexity(gray)
                        
                        # 初始总分
                        total_score = score + complexity_score
                        
                        # 生成唯一ID
                        frame_id = f"scene_{frame_idx}"
                        
                        # 将帧添加到候选列表
                        candidates.append({
                            'id': frame_id,
                            'frame': frame.copy(),
                            'score': total_score,
                            'time': current_time,
                            'timestamp': timestamp,
                            'is_scene_change': True,
                        })
                        
                        logger.info(f"检测到场景切换 - 时间: {timestamp}, 初始分数: {total_score:.2f}")
                
                prev_hist = hist
            else:
                prev_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
            
            # 内容稳定性分析
            frames_buffer.append((frame.copy(), gray, current_time, timestamp))
            buffer_size = len(frames_buffer)
            
            # 检查缓冲区中的帧是否稳定
            if buffer_size >= self.stable_duration * self.sample_rate:
                is_stable = self._check_stability(frames_buffer)
                
                if is_stable and stable_start is None:
                    stable_start = buffer_size - self.stable_duration * self.sample_rate
                
                # 如果不再稳定或遇到场景切换，处理稳定序列
                if (not is_stable or scene_change) and stable_start is not None:
                    # 选择稳定序列的中间帧
                    mid_idx = stable_start + (buffer_size - stable_start) // 2
                    if mid_idx < len(frames_buffer):
                        stable_frame, stable_gray, stable_time, stable_timestamp = frames_buffer[mid_idx]
                        
                        # 视觉复杂度分析
                        complexity_score = self._analyze_complexity(stable_gray)
                        
                        # 计算总分 - 稳定序列基础分
                        total_score = 15 + complexity_score
                        
                        # 生成唯一ID
                        frame_id = f"stable_{frame_idx}_{mid_idx}"
                        
                        # 将帧添加到候选列表
                        candidates.append({
                            'id': frame_id,
                            'frame': stable_frame,
                            'score': total_score,
                            'time': stable_time,
                            'timestamp': stable_timestamp,
                            'is_scene_change': False,
                        })
                        
                        logger.info(f"检测到稳定内容 - 时间: {stable_timestamp}, 初始分数: {total_score:.2f}")
                    
                    stable_start = None
                
                # 移除旧帧，保持缓冲区大小合理
                if len(frames_buffer) > self.stable_duration * self.sample_rate * 2:
                    frames_buffer.pop(0)
            
            prev_frame = gray
            frame_idx += 1
            
            # 显示进度
            if frame_idx % (sample_interval * 10) == 0:
                progress = (frame_idx / self.frame_count) * 100
                logger.info(f"处理进度: {progress:.1f}%")
        
        self.cap.release()
        
        # 对候选帧进行排序和去重，并存储
        self._select_and_save_frames(candidates)
        
        # 保存元数据
        self._save_metadata()
        
        # 生成HTML报告
        self._generate_html_report()
        
        elapsed_time = time.time() - start_time
        logger.info(f"关键帧提取完成! 耗时: {elapsed_time:.2f}秒")
        return self.candidate_frames
    
    def _check_stability(self, frames_buffer: List) -> bool:
        """
        检查一系列帧是否稳定
        
        Args:
            frames_buffer: 帧缓冲区
            
        Returns:
            帧序列是否稳定
        """
        if len(frames_buffer) < 2:
            return False
        
        # 比较第一帧和最后一帧
        _, first_gray, _, _ = frames_buffer[0]
        _, last_gray, _, _ = frames_buffer[-1]
        
        # 计算SSIM
        similarity = ssim(first_gray, last_gray)
        
        # 如果相似度高于阈值，认为是稳定的
        return similarity > 0.8
    
    def _analyze_complexity(self, gray: np.ndarray) -> float:
        """
        分析图像的视觉复杂度
        
        Args:
            gray: 灰度图像
            
        Returns:
            复杂度分数
        """
        # 1. 使用Canny边缘检测计算边缘密度
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # 2. 计算图像熵
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]  # 过滤掉0概率的像素值
        entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
        
        # 3. 检测矩形区域和线条 - 适用于教学内容如幻灯片、白板等
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 区域分析分数
        region_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 忽略太小的区域
                # 检查是否为矩形区域
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                if len(approx) == 4:  # 矩形有4个角点
                    region_score += min(20, area / 10000)
                    
                # 检查轮廓的纵横比 - 文本和图表通常有特定的纵横比
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5:
                    region_score += min(10, area / 5000)
        
        # 4. 表格检测 - 查找水平和垂直线
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        h_count = 0 if h_lines is None else len(h_lines)
        v_count = 0 if v_lines is None else len(v_lines)
        
        # 根据线数量评分
        table_score = 0
        if h_count >= 3 and v_count >= 3:
            table_score = 40  # 很可能是表格
        elif h_count >= 2 and v_count >= 2:
            table_score = 30  # 可能是表格
        elif h_count >= 1 and v_count >= 1:
            table_score = 15  # 可能有表格元素
        
        # 5. 计算Laplacian得分 - 检测图像清晰度/聚焦度
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_score = min(20, np.var(laplacian) / 100)
        
        # 综合分数 - 调整各项分数的权重
        complexity_score = (
            edge_density * 40 +        # 边缘密度
            entropy * 10 +             # 熵
            min(30, region_score) +    # 区域分析
            table_score +              # 表格检测
            laplacian_score            # 清晰度
        )
        
        # 限制最大分数
        return min(60, complexity_score * 0.3)
    
    def _get_image_hash(self, frame: np.ndarray) -> str:
        """
        计算图像的感知哈希，用于图像去重
        
        Args:
            frame: 图像
            
        Returns:
            图像哈希值
        """
        # 调整大小为8x8
        resized = cv2.resize(frame, (8, 8), interpolation=cv2.INTER_AREA)
        
        # 转换为灰度图
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) > 2 else resized
        
        # 计算均值
        avg_value = np.mean(gray)
        
        # 生成哈希
        hash_str = ''
        for i in range(8):
            for j in range(8):
                hash_str += '1' if gray[i, j] > avg_value else '0'
        
        return hash_str
    
    def _select_and_save_frames(self, candidates: List[Dict[str, Any]]) -> None:
        """
        选择并保存最佳候选帧
        
        Args:
            candidates: 候选帧列表
        """
        if not candidates:
            logger.warning("没有找到关键帧！")
            return
        
        # 按分数排序，降序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 最低分数阈值
        min_score = 25
        
        # 最小时间间隔（秒）
        min_time_interval = 2.0
        
        # 已保存的帧的时间戳
        saved_times = []
        
        # 处理候选帧
        for idx, candidate in enumerate(candidates):
            # 如果分数太低，跳过
            if candidate['score'] < min_score:
                continue
            
            # 检查时间间隔
            too_close = False
            for saved_time in saved_times:
                if abs(candidate['time'] - saved_time) < min_time_interval:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # 计算图像哈希用于去重
            img_hash = self._get_image_hash(candidate['frame'])
            
            # 如果图像与之前保存的非常相似，跳过
            hash_distance_threshold = 5  # 允许的哈希差异
            for saved_hash in self.saved_hashes:
                # 计算汉明距离
                distance = sum(c1 != c2 for c1, c2 in zip(img_hash, saved_hash))
                if distance < hash_distance_threshold:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # 确定文件名，使用时间戳作为主要排序依据
            seconds = int(candidate['time'])
            timestamp_str = candidate['timestamp'].replace(':', '-')
            frame_type = "scene" if candidate['is_scene_change'] else "stable"
                
            # 完整文件名
            filename = f"{seconds:06d}_{timestamp_str}_{frame_type}.jpg"
            file_path = os.path.join(self.frames_dir, filename)
            
            # 保存图像
            cv2.imwrite(file_path, candidate['frame'])
            
            # 更新保存的图像信息
            self.saved_hashes.add(img_hash)
            saved_times.append(candidate['time'])
            
            # 保存关键帧信息
            frame_info = {
                'id': idx + 1,
                'filename': filename,
                'relative_path': os.path.join("frames", filename),
                'time': candidate['time'],
                'timestamp': candidate['timestamp'],
                'seconds': seconds,
                'score': candidate['score'],
                'is_scene_change': candidate['is_scene_change'],
                'is_valuable': False  # 默认值，将在图像理解模块中更新
            }
            
            self.candidate_frames.append(frame_info)
            
            # 更新元数据
            self.metadata["frames"].append(frame_info)
            
            logger.info(f"保存帧: {file_path}")
        
        logger.info(f"共保存了 {len(self.candidate_frames)} 个关键帧")
    
    def update_frame_value(self, frame_id: int, is_valuable: bool, description: str = "") -> None:
        """
        更新帧的价值评估结果
        
        Args:
            frame_id: 帧ID
            is_valuable: 是否有价值
            description: 内容描述
        """
        # 更新候选帧列表
        for frame in self.candidate_frames:
            if frame['id'] == frame_id:
                frame['is_valuable'] = is_valuable
                frame['content_description'] = description
                
                # 如果有价值，复制到valuable_frames目录
                if is_valuable:
                    src_path = os.path.join(self.frames_dir, frame['filename'])
                    dst_path = os.path.join(self.valuable_frames_dir, frame['filename'])
                    if os.path.exists(src_path) and not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                break
        
        # 更新元数据
        for frame in self.metadata['frames']:
            if frame['id'] == frame_id:
                frame['is_valuable'] = is_valuable
                frame['content_description'] = description
                break
                
        # 保存更新后的元数据
        self._save_metadata()
    
    def _save_metadata(self) -> None:
        """保存元数据到JSON文件"""
        metadata_file = os.path.join(self.video_output_dir, "metadata.json")
        
        # 添加总结信息
        total_frames = len(self.metadata["frames"])
        valuable_frames = sum(1 for frame in self.metadata["frames"] if frame.get('is_valuable', False))
        scene_changes = sum(1 for frame in self.metadata["frames"] if frame.get('is_scene_change', True))
        stable_frames = sum(1 for frame in self.metadata["frames"] if not frame.get('is_scene_change', False))
        
        self.metadata["summary"] = {
            "total_frames": total_frames,
            "valuable_frames": valuable_frames,
            "scene_changes": scene_changes,
            "stable_frames": stable_frames,
            "extraction_completed": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 排序帧，确保时间顺序
        self.metadata["frames"].sort(key=lambda x: x['time'])
        
        # 保存元数据
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"已保存元数据到: {metadata_file}")
        
        # 生成简化版元数据(供笔记生成使用)
        simplified_metadata = {
            "video_title": self.video_name,
            "duration": self._format_duration(self.duration),
            "frames": []
        }
        
        # 只包含有价值的帧且按时间排序
        valuable_frames = []
        for frame in self.metadata["frames"]:
            if frame.get('is_valuable', False):
                valuable_frames.append({
                    "id": frame['id'],
                    "timestamp": frame['timestamp'],
                    "seconds": frame['seconds'],
                    "filename": frame['filename'],
                    "is_scene_change": frame['is_scene_change'],
                    "content_description": frame.get('content_description', "")
                })
        
        # 对有价值的帧排序
        valuable_frames.sort(key=lambda x: x['seconds'])
        simplified_metadata["frames"] = valuable_frames
        
        # 保存简化版元数据
        simplified_file = os.path.join(self.valuable_frames_dir, "frames_for_notes.json")
        with open(simplified_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"已保存简化版元数据到: {simplified_file}")
    
    def _format_duration(self, seconds: float) -> str:
        """
        格式化视频时长
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时长字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}小时{minutes}分钟{secs}秒"
        else:
            return f"{minutes}分钟{secs}秒"
    
    def _generate_html_report(self) -> None:
        """生成HTML报告"""
        html_report = os.path.join(self.video_output_dir, "report.html")
        
        with open(html_report, 'w', encoding='utf-8') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>关键帧提取报告 - {self.video_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .frame {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                    .frame img {{ max-width: 80%; max-height: 500px; }}
                    .info {{ margin-top: 10px; }}
                    .description {{ margin-top: 10px; background-color: #e6f7ff; padding: 10px; border-radius: 5px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; margin-bottom: 20px; }}
                    .container {{ display: flex; flex-wrap: wrap; }}
                    .filter-btn {{ margin: 5px; padding: 8px 15px; background-color: #4CAF50; color: white; 
                                  border: none; border-radius: 4px; cursor: pointer; }}
                    .filter-btn:hover {{ background-color: #45a049; }}
                    .valuable {{ border-left: 5px solid #4CAF50; }}
                    .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
                    .tab button {{ background-color: inherit; float: left; border: none; outline: none;
                                  cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
                    .tab button:hover {{ background-color: #ddd; }}
                    .tab button.active {{ background-color: #ccc; }}
                    .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }}
                </style>
                <script>
                    function filterFrames(type) {{
                        let frames = document.getElementsByClassName('frame');
                        for (let i = 0; i < frames.length; i++) {{
                            if (type === 'all') {{
                                frames[i].style.display = 'block';
                            }} else if (type === 'scene' && frames[i].classList.contains('scene-change')) {{
                                frames[i].style.display = 'block';
                            }} else if (type === 'stable' && frames[i].classList.contains('stable-content')) {{
                                frames[i].style.display = 'block';
                            }} else if (type === 'valuable' && frames[i].classList.contains('is-valuable')) {{
                                frames[i].style.display = 'block';
                            }} else {{
                                frames[i].style.display = 'none';
                            }}
                        }}
                    }}
                    
                    function openTab(evt, tabName) {{
                        var i, tabcontent, tablinks;
                        tabcontent = document.getElementsByClassName("tabcontent");
                        for (i = 0; i < tabcontent.length; i++) {{
                            tabcontent[i].style.display = "none";
                        }}
                        tablinks = document.getElementsByClassName("tablinks");
                        for (i = 0; i < tablinks.length; i++) {{
                            tablinks[i].className = tablinks[i].className.replace(" active", "");
                        }}
                        document.getElementById(tabName).style.display = "block";
                        evt.currentTarget.className += " active";
                    }}
                </script>
            </head>
            <body>
                <div class="header">
                    <h1>视频关键帧提取报告</h1>
                    <p>视频文件: {self.video_path}</p>
                    <p>提取的关键帧数量: {len(self.candidate_frames)}</p>
                    
                    <div class="tab">
                        <button class="tablinks active" onclick="openTab(event, 'Overview')">概览</button>
                        <button class="tablinks" onclick="openTab(event, 'AllFrames')">全部帧</button>
                        <button class="tablinks" onclick="openTab(event, 'ValuableFrames')">有价值帧</button>
                    </div>
                    
                    <div id="Overview" class="tabcontent" style="display: block;">
                        <h2>视频内容概览</h2>
                        <p>视频总时长: {self._format_duration(self.duration)}</p>
                        <p>提取的关键帧总数: {len(self.candidate_frames)}</p>
                        <p>场景切换数: {sum(1 for frame in self.candidate_frames if frame.get('is_scene_change', False))}</p>
                        <p>稳定内容数: {sum(1 for frame in self.candidate_frames if not frame.get('is_scene_change', True))}</p>
                        <p>评估为有价值的帧数: {sum(1 for frame in self.candidate_frames if frame.get('is_valuable', False))}</p>
                        <h3>处理参数</h3>
                        <ul>
                            <li>采样率: {self.sample_rate} 帧/秒</li>
                            <li>稳定持续时间: {self.stable_duration} 秒</li>
                            <li>场景切换阈值: {self.scene_threshold}</li>
                        </ul>
                    </div>
                    
                    <div id="AllFrames" class="tabcontent">
                        <h2>全部关键帧</h2>
                        <div>
                            <button class="filter-btn" onclick="filterFrames('all')">显示全部</button>
                            <button class="filter-btn" onclick="filterFrames('scene')">仅场景切换</button>
                            <button class="filter-btn" onclick="filterFrames('stable')">仅稳定内容</button>
                            <button class="filter-btn" onclick="filterFrames('valuable')">仅有价值</button>
                        </div>
                        <div class="container">
            """)
            
            # 添加所有帧
            for frame in sorted(self.candidate_frames, key=lambda x: x['time']):
                frame_type = "场景切换" if frame.get('is_scene_change', False) else "稳定内容"
                frame_class = "scene-change" if frame.get('is_scene_change', False) else "stable-content"
                
                # 检测是否有价值
                valuable_class = ""
                if frame.get('is_valuable', False):
                    frame_class += " is-valuable"
                    valuable_class = "valuable"
                
                rel_path = os.path.join("frames", frame['filename'])
                
                f.write(f"""
                <div class="frame {frame_class} {valuable_class}">
                    <h2>帧 #{frame['id']} - {frame['timestamp']}</h2>
                    <img src="{rel_path}" alt="关键帧">
                    <div class="info">
                        <p>时间: {frame['timestamp']} | 分数: {frame['score']:.2f} | 类型: {frame_type}</p>
                    </div>
                """)
                
                if 'content_description' in frame and frame.get('is_valuable', False):
                    f.write(f"""
                    <div class="description">
                        <h3>内容描述:</h3>
                        <p>{frame['content_description']}</p>
                    </div>
                    """)
                
                f.write(f"""
                </div>
                """)
            
            f.write("""
                        </div>
                    </div>
                    
                    <div id="ValuableFrames" class="tabcontent">
                        <h2>有价值的帧</h2>
                        <div class="container">
            """)
            
            # 添加有价值的帧
            valuable_frames = [frame for frame in self.candidate_frames if frame.get('is_valuable', False)]
            for frame in sorted(valuable_frames, key=lambda x: x['time']):
                rel_path = os.path.join("frames", frame['filename'])
                
                f.write(f"""
                <div class="frame valuable">
                    <h2>帧 #{frame['id']} - {frame['timestamp']}</h2>
                    <img src="{rel_path}" alt="关键帧">
                    <div class="info">
                        <p>时间: {frame['timestamp']} | 分数: {frame['score']:.2f}</p>
                    </div>
                """)
                
                if 'content_description' in frame:
                    f.write(f"""
                    <div class="description">
                        <h3>内容描述:</h3>
                        <p>{frame['content_description']}</p>
                    </div>
                    """)
                
                f.write(f"""
                </div>
                """)
            
            if not valuable_frames:
                f.write("<p>没有评估为有价值的帧。</p>")
            
            f.write("""
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """)
        
        logger.info(f"已生成HTML报告: {html_report}")
    
    def get_all_frames(self) -> List[Dict[str, Any]]:
        """
        获取所有已提取的帧
        
        Returns:
            帧列表
        """
        return self.candidate_frames
    
    def get_valuable_frames(self) -> List[Dict[str, Any]]:
        """
        获取所有评估为有价值的帧
        
        Returns:
            有价值的帧列表
        """
        return [frame for frame in self.candidate_frames if frame.get('is_valuable', False)]
    
    def get_frame_by_id(self, frame_id: int) -> Optional[Dict[str, Any]]:
        """
        通过ID获取帧
        
        Args:
            frame_id: 帧ID
            
        Returns:
            帧信息
        """
        for frame in self.candidate_frames:
            if frame['id'] == frame_id:
                return frame
        return None


# 便捷函数
def extract_frames(
    video_path: str, 
    output_dir: str = "output", 
    sample_rate: int = 1, 
    stable_duration: int = 3, 
    scene_threshold: float = 0.3
) -> FrameExtractor:
    """
    提取视频关键帧的便捷函数
    
    Args:
        video_path: 输入视频的路径
        output_dir: 保存图像的主目录
        sample_rate: 每秒采样的帧数
        stable_duration: 定义内容稳定的最小持续秒数
        scene_threshold: 场景切换检测的阈值 (0-1)
        
    Returns:
        FrameExtractor实例
    """
    extractor = FrameExtractor(
        video_path, 
        output_dir,
        sample_rate,
        stable_duration,
        scene_threshold
    )
    
    extractor.extract_frames()
    
    print("\n处理完成! 提取的关键帧已保存到以下目录:")
    print(f"- 所有帧: {os.path.join(output_dir, extractor.video_name, 'frames')}")
    print(f"- 有价值的帧: {os.path.join(output_dir, extractor.video_name, 'valuable_frames')}")
    print(f"\n可以查看报告: {os.path.join(output_dir, extractor.video_name, 'report.html')}")
    
    return extractor


if __name__ == "__main__":
    import argparse
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='从课程视频中提取关键帧')
    parser.add_argument('video_path', type=str, help='输入视频的路径')
    parser.add_argument('--output_dir', type=str, default='output', help='保存图像的主目录')
    parser.add_argument('--sample_rate', type=int, default=1, help='每秒采样的帧数')
    parser.add_argument('--stable_duration', type=int, default=3, help='内容稳定的最小持续秒数')
    parser.add_argument('--scene_threshold', type=float, default=0.3, help='场景切换检测的阈值 (0-1)')
    
    args = parser.parse_args()
    
    extract_frames(
        args.video_path, 
        args.output_dir,
        args.sample_rate,
        args.stable_duration,
        args.scene_threshold
    )