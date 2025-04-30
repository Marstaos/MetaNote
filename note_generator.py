"""
笔记生成模块
负责将视频转录和关键帧整合为Markdown笔记
"""

import os
import json
import logging
import time
import shutil
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# 尝试导入OpenAI库 (用于千问API)
try:
    from openai import OpenAI
    has_openai = True
except ImportError:
    has_openai = False

logger = logging.getLogger(__name__)

class NoteGenerator:
    """笔记生成器基类"""
    
    def __init__(self):
        """初始化笔记生成器"""
        pass
    
    def generate_notes(
        self, 
        transcript: str, 
        frames_info: List[Dict[str, Any]], 
        output_path: str,
        title: Optional[str] = None,
        summary: Optional[str] = None
    ) -> str:
        """
        生成笔记
        
        Args:
            transcript: 视频转录文本
            frames_info: 关键帧信息列表
            output_path: 输出文件路径
            title: 笔记标题
            summary: 内容概要
            
        Returns:
            生成的笔记内容
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def process_frame_extractor_and_transcript(
        self, 
        frame_extractor, 
        transcript: Dict[str, Any],
        output_dir: Optional[str] = None
    ) -> str:
        """
        处理帧提取器和转录结果生成笔记
        
        Args:
            frame_extractor: FrameExtractor实例
            transcript: 转录结果
            output_dir: 输出目录
            
        Returns:
            生成的笔记文件路径
        """
        # 获取视频标题
        video_name = frame_extractor.video_name
        
        # 确定输出目录
        if output_dir is None:
            output_dir = frame_extractor.video_output_dir
        
        # 获取有价值的帧
        valuable_frames = frame_extractor.get_valuable_frames()
        if not valuable_frames:
            logger.warning("没有找到有价值的帧，将使用所有帧")
            # 使用所有帧但按得分排序
            all_frames = sorted(frame_extractor.get_all_frames(), key=lambda x: x.get('score', 0), reverse=True)
            valuable_frames = all_frames[:min(15, len(all_frames))]  # 最多使用15帧
        
        # 准备帧信息，包括绝对和相对路径
        frames_info = []
        for frame in sorted(valuable_frames, key=lambda x: x['time']):
            frame_path = os.path.join(frame_extractor.frames_dir, frame['filename'])
            if os.path.exists(frame_path):
                frames_info.append({
                    'id': frame['id'],
                    'file_path': frame_path,
                    'relative_path': os.path.join("frames", frame['filename']),
                    'timestamp': frame['timestamp'],
                    'seconds': frame['seconds'],
                    'is_scene_change': frame.get('is_scene_change', False),
                    'description': frame.get('content_description', ""),
                    'score': frame.get('score', 0)
                })
        
        # 准备转录文本
        transcription_text = self._process_transcript(transcript)
        
        # 确定输出文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"notes_{video_name}_{timestamp}.md")
        
        # 生成笔记
        notes_content = self.generate_notes(
            transcription_text,
            frames_info,
            output_path,
            title=video_name
        )
        
        return output_path
    
    def _process_transcript(self, transcript: Any) -> str:
        """
        处理转录结果为纯文本
        
        Args:
            transcript: 转录结果
            
        Returns:
            处理后的文本
        """
        # 处理ASR模型的结果格式
        if isinstance(transcript, dict):
            # 尝试从字典中提取结果
            if 'result' in transcript:
                result = transcript['result']
                
                # 处理列表类型结果
                if isinstance(result, list):
                    # 检查是否为字典列表
                    if all(isinstance(item, dict) for item in result):
                        # 检查列表中的字典是否包含文本
                        if all('text' in item for item in result):
                            return "\n".join(item['text'] for item in result)
                        
                    # 尝试直接连接列表项
                    return "\n".join(str(item) for item in result)
                    
                # 处理字典类型结果
                elif isinstance(result, dict):
                    if 'text' in result:
                        return result['text']
                    else:
                        return json.dumps(result, ensure_ascii=False, indent=2)
                
                # 处理字符串类型结果
                elif isinstance(result, str):
                    return result
                
                # 其他类型，转为字符串
                else:
                    return str(result)
            
            # 直接返回 JSON 字符串
            return json.dumps(transcript, ensure_ascii=False, indent=2)
        
        # 如果是列表，尝试合并
        elif isinstance(transcript, list):
            return "\n".join(str(item) for item in transcript)
        
        # 如果是字符串，直接返回
        elif isinstance(transcript, str):
            return transcript
        
        # 其他类型，转为字符串
        else:
            return str(transcript)
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """
        从文本中提取关键概念
        
        Args:
            text: 输入文本
            
        Returns:
            关键概念列表
        """
        concepts = []
        # 分割成句子
        sentences = re.split(r'[。！？.!?；;]', text)
        # 关键词列表
        keywords = [
            "重要", "关键", "核心", "要点", "记住", "注意", "关注", "必须", "应该", "总结", 
            "概念", "定义", "特点", "优点", "缺点", "分类", "步骤", "方法", "原理", "理论",
            "公式", "算法", "模型", "规则", "标准", "原则", "策略"
        ]
        
        # 找出包含关键词的句子
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # 忽略过短的句子
                continue
                
            # 检查句子是否包含关键词
            if any(keyword in sentence for keyword in keywords):
                if sentence not in concepts:  # 避免重复
                    concepts.append(sentence)
            
            # 检查是否是定义句式（通常包含"是"、"指"、"称为"等词）
            definition_patterns = [
                r".*是.*的.*",
                r".*指.*",
                r".*定义为.*",
                r".*称为.*",
                r".*叫做.*"
            ]
            if any(re.match(pattern, sentence) for pattern in definition_patterns):
                if sentence not in concepts:
                    concepts.append(sentence)
        
        # 如果提取的概念太少，选择一些较长的句子作为补充
        if len(concepts) < 5:
            longer_sentences = sorted([s.strip() for s in sentences if len(s.strip()) > 20], 
                                     key=len, reverse=True)
            for s in longer_sentences[:5-len(concepts)]:
                if s not in concepts:
                    concepts.append(s)
        
        return concepts
    
    def _extract_possible_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        从文本中提取可能的章节标题和内容
        
        Args:
            text: 输入文本
            
        Returns:
            章节标题和内容的元组列表
        """
        # 分段
        paragraphs = text.split('\n\n')
        
        # 如果段落太少，按句子分割
        if len(paragraphs) < 3:
            sentences = re.split(r'[。！？.!?]', text)
            # 根据句子长度和关键词重新组织段落
            paragraphs = []
            current_paragraph = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                current_paragraph.append(sentence)
                
                # 如果当前句子很短或包含特定词语，可能是一个段落的结束
                if len(sentence) < 15 or any(kw in sentence for kw in ["总结", "因此", "所以", "最后"]):
                    if current_paragraph:
                        paragraphs.append("。".join(current_paragraph))
                        current_paragraph = []
            
            # 添加最后一个段落
            if current_paragraph:
                paragraphs.append("。".join(current_paragraph))
        
        # 尝试提取章节
        sections = []
        current_title = None
        current_content = []
        
        # 关键词集合，用于识别可能的标题
        title_keywords = [
            "介绍", "简介", "概述", "背景", "定义", "原理", "方法", "步骤", "过程", "特点",
            "分类", "类型", "优点", "缺点", "应用", "实现", "总结", "结论", "讨论"
        ]
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            first_sentence = re.split(r'[。！？.!?]', para)[0].strip()
            
            # 判断段落的第一句是否可能是标题
            is_possible_title = (
                len(first_sentence) < 30 and 
                any(keyword in first_sentence for keyword in title_keywords)
            )
            
            if is_possible_title:
                # 保存之前的章节
                if current_title and current_content:
                    sections.append((current_title, "\n\n".join(current_content)))
                
                # 开始新章节
                current_title = first_sentence
                current_content = [para[len(first_sentence):].strip()]
            else:
                # 继续当前章节
                if not current_title:
                    # 如果还没有章节标题，创建一个
                    if len(sections) == 0:
                        current_title = "课程简介"
                    else:
                        current_title = f"第 {len(sections)+1} 部分"
                
                current_content.append(para)
        
        # 添加最后一个章节
        if current_title and current_content:
            sections.append((current_title, "\n\n".join(current_content)))
        
        # 如果没有提取到章节，创建默认章节
        if not sections:
            sections = [
                ("课程简介", text[:min(500, len(text))]),
                ("主要内容", text[min(500, len(text)):])
            ]
        
        return sections
    
    def _get_template(self) -> str:
        """
        获取Markdown笔记模板
        
        Returns:
            模板内容
        """
        return """# {title}

## 课程概要

{summary}

{content}

## 重要概念与关键点

{key_points}

## 总结

{conclusion}

---

*这些笔记由MetaNote自动生成于 {date}*
"""


class QwenNoteGenerator(NoteGenerator):
    """使用阿里云千问大模型生成笔记"""
    
    def __init__(
        self, 
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-plus",
        system_prompt: Optional[str] = None
    ):
        """
        初始化千问笔记生成器
        
        Args:
            api_key: 千问API密钥
            base_url: API基础URL
            model: 使用的模型
            system_prompt: 系统提示语
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = None
        self.initialized = False
        
        # 设置系统提示语
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = (
                "你是一位专业的教育内容编辑和学术笔记专家，擅长将课程视频的文字记录和图像整理成结构化、详尽的学习笔记。"
                "你的任务是根据提供的视频转录文本和关键帧信息，创建一份专业、系统化的Markdown格式笔记。"
                "笔记应该包含：\n"
                "1. 清晰的标题、副标题和层次结构\n"
                "2. 完整保留视频中的核心知识点和关键信息，不要过度精简\n"
                "3. 使用学术规范的语言和格式，确保术语使用准确\n"
                "4. 将图像与相关文本内容紧密结合，提供对图像内容的专业解释\n"
                "5. 整理关键概念、定义和重要公式，使它们易于理解和记忆\n"
                "6. 添加小结或总结部分，帮助读者把握整体内容\n"
                "7. 保持内容的完整性和连贯性，确保笔记即使脱离视频也能独立理解\n"
                "注重笔记的学术严谨性和专业性，同时保证内容的完整保留和系统化组织。"
            )
        
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
                    {"role": "system", "content": "You are a helpful assistant."},
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
    
    def generate_notes(
        self, 
        transcript: str, 
        frames_info: List[Dict[str, Any]], 
        output_path: str,
        title: Optional[str] = None,
        summary: Optional[str] = None
    ) -> str:
        """
        生成笔记
        
        Args:
            transcript: 视频转录文本
            frames_info: 关键帧信息列表
            output_path: 输出文件路径
            title: 笔记标题
            summary: 内容概要
            
        Returns:
            生成的笔记内容
        """
        if not has_openai:
            return self._generate_basic_notes(transcript, frames_info, output_path, title, summary)
            
        if not self.initialized and not self.initialize():
            logger.warning("千问API未初始化，将使用基本模板生成笔记")
            return self._generate_basic_notes(transcript, frames_info, output_path, title, summary)
        
        # 准备图像目录
        output_dir = os.path.dirname(output_path)
        img_dir = os.path.join(output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        # 复制图像到输出目录
        for i, frame in enumerate(frames_info):
            img_path = frame.get('file_path')
            if img_path and os.path.exists(img_path):
                # 创建新的图像名称
                new_name = f"image_{i+1:02d}.jpg"
                new_path = os.path.join(img_dir, new_name)
                
                # 复制图像
                shutil.copy2(img_path, new_path)
                
                # 更新图像路径
                frame['md_path'] = f"./images/{new_name}"
                frame['md_ref'] = f"![图片 {i+1} - {frame.get('timestamp', '')}]({frame['md_path']})"
        
        # 准备输入
        # 1. 预处理转录文本，截取前12000个字符以保留更多内容，但避免超过模型窗口限制
        if len(transcript) > 12000:
            transcript_text = transcript[:12000] + "...(转录文本过长，已截断)"
            logger.warning(f"转录文本过长 ({len(transcript)} 字符)，已截断至12000字符")
        else:
            transcript_text = transcript
        
        # 2. 准备关键帧信息
        frames_text = ""
        for i, frame in enumerate(frames_info):
            frame_desc = [
                f"图片{i+1}：",
                f"- 时间戳：{frame.get('timestamp', 'unknown')}",
                f"- 类型：{'场景切换' if frame.get('is_scene_change', False) else '稳定内容'}",
            ]
            
            if 'description' in frame and frame['description']:
                frame_desc.append(f"- 内容描述：{frame['description']}")
            
            if 'md_ref' in frame:
                frame_desc.append(f"- Markdown引用：{frame['md_ref']}")
            
            frames_text += "\n".join(frame_desc) + "\n\n"
        
        # 组装提示文本
        prompt = f"""我需要你根据以下视频转录文本和关键帧信息，生成一份专业、详细的学术Markdown笔记。

## 视频标题
{title or "未命名课程"}

## 视频转录文本
{transcript_text}

## 关键帧信息
{frames_text}

请生成一份完整且专业的Markdown笔记，要求如下：

1. 笔记应包含清晰的标题结构（使用Markdown标题层级：#、##、###等）
2. **保留视频中的所有关键信息和重要细节**，不要过度精简内容
3. 将内容组织成有逻辑的章节，每个章节都应该完整详细
4. 根据视频内容的复杂度，分为适当数量的章节（通常4-7个）
5. 适当位置插入图片引用（使用提供的Markdown引用语法）
6. 在每个图片下方提供详细的图片内容解释，与周围文本内容紧密结合
7. 包含"重要概念与关键点"部分，整理视频中的核心概念和关键要点
8. 添加总结部分，概括视频的主要内容和学习目标
9. 使用专业术语和学术语言，确保专业性和准确性
10. 通篇结构清晰，层次分明，便于阅读和理解

请直接给出完整的Markdown文本，不要解释你的处理过程。确保生成的笔记内容全面、系统，能够作为学习参考资料使用。"""

        try:
            # 调用模型生成笔记
            logger.info("正在使用千问模型生成笔记...")
            start_time = time.time()
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # 提取生成的内容
            generated_notes = completion.choices[0].message.content
            
            end_time = time.time()
            logger.info(f"笔记生成完成，耗时：{end_time - start_time:.2f}秒")
            
            # 保存笔记到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generated_notes)
            
            logger.info(f"笔记已保存到：{output_path}")
            
            return generated_notes
            
        except Exception as e:
            logger.error(f"生成笔记时出错: {str(e)}")
            return self._generate_basic_notes(transcript, frames_info, output_path, title, summary)
    
    def _generate_basic_notes(
        self, 
        transcript: str, 
        frames_info: List[Dict[str, Any]], 
        output_path: str,
        title: Optional[str] = None,
        summary: Optional[str] = None
    ) -> str:
        """
        当API调用失败时，使用基本模板生成笔记
        
        Args:
            transcript: 视频转录文本
            frames_info: 关键帧信息列表
            output_path: 输出文件路径
            title: 笔记标题
            summary: 内容概要
            
        Returns:
            生成的笔记内容
        """
        logger.info("使用基本模板生成详细笔记...")
        
        # 准备图像目录
        output_dir = os.path.dirname(output_path)
        img_dir = os.path.join(output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        # 复制图像到输出目录
        for i, frame in enumerate(frames_info):
            img_path = frame.get('file_path')
            if img_path and os.path.exists(img_path):
                # 创建新的图像名称
                new_name = f"image_{i+1:02d}.jpg"
                new_path = os.path.join(img_dir, new_name)
                
                # 复制图像
                shutil.copy2(img_path, new_path)
                
                # 更新图像路径
                frame['md_path'] = f"./images/{new_name}"
                frame['md_ref'] = f"![图片 {i+1} - {frame.get('timestamp', '')}]({frame['md_path']})"
        
        # 获取模板
        template = self._get_template()
        
        # 提取可能的标题
        if not title:
            title = "课程笔记"
            # 从转录文本中提取可能的标题
            lines = transcript.split('\n')
            if lines and len(lines[0]) < 100:
                title = lines[0]
        
        # 提取或生成摘要
        if not summary:
            # 尝试从转录文本的前几句话中提取摘要
            sentences = re.split(r'[。！？.!?]', transcript)
            summary_sentences = []
            for s in sentences[:5]:  # 取前5个句子
                if len(s.strip()) > 10:  # 忽略太短的句子
                    summary_sentences.append(s.strip())
            
            if summary_sentences:
                summary = "。".join(summary_sentences) + "。"
            else:
                # 退化情况：使用前300个字符
                summary = transcript[:300] + "..." if len(transcript) > 300 else transcript
        
        # 提取关键概念
        key_concepts = self._extract_key_concepts(transcript)
        
        # 提取可能的章节
        sections = self._extract_possible_sections(transcript)
        
        # 基于章节时间结构，将图像分配到相应章节
        # 首先，将转录文本分为几个时间段
        total_frames = len(frames_info)
        sections_with_images = []
        
        if total_frames > 0:
            # 计算每个章节应分配的图像数量
            images_per_section = defaultdict(list)
            section_count = len(sections)
            
            # 更均匀地分配图像
            for i, frame in enumerate(frames_info):
                # 确定此图像应属于哪个章节 (基于时间比例)
                section_idx = min(int(i * section_count / total_frames), section_count - 1)
                images_per_section[section_idx].append(frame)
            
            # 构建包含图像的章节内容
            for i, (section_title, section_content) in enumerate(sections):
                section_images = images_per_section[i]
                
                # 添加章节标题和内容开头
                section_text = f"## {section_title}\n\n"
                
                # 分段以插入图像
                paragraphs = section_content.split("\n\n")
                
                # 如果此章节有图像
                if section_images:
                    # 计算每个图像之间应该有多少段落
                    paras_per_image = max(1, len(paragraphs) // (len(section_images) + 1))
                    
                    # 插入图像和内容
                    current_para = 0
                    for j, image in enumerate(section_images):
                        # 添加一些段落
                        next_para = min(current_para + paras_per_image, len(paragraphs))
                        section_text += "\n\n".join(paragraphs[current_para:next_para]) + "\n\n"
                        current_para = next_para
                        
                        # 添加图像
                        section_text += f"{image['md_ref']}\n\n"
                        
                        # 添加图像描述
                        if image.get('description'):
                            section_text += f"*{image['description']}*\n\n"
                    
                    # 添加剩余段落
                    if current_para < len(paragraphs):
                        section_text += "\n\n".join(paragraphs[current_para:]) + "\n\n"
                else:
                    # 没有图像，直接添加所有段落
                    section_text += section_content + "\n\n"
                
                sections_with_images.append(section_text)
        else:
            # 没有图像，直接使用原始章节
            sections_with_images = [f"## {title}\n\n{content}\n\n" for title, content in sections]
        
        # 组装内容
        content = "\n".join(sections_with_images)
        
        # 格式化关键概念
        key_points_text = ""
        if key_concepts:
            for i, concept in enumerate(key_concepts):
                key_points_text += f"{i+1}. {concept}\n\n"
        else:
            key_points_text = "视频中未能提取到明确的关键概念。\n\n"
        
        # 提取总结
        # 尝试从转录文本的最后一部分提取总结
        conclusion_text = ""
        last_paragraphs = transcript.split('\n\n')[-3:]  # 最后三段
        for para in last_paragraphs:
            if any(keyword in para for keyword in ["总结", "小结", "结论", "总的来说", "总而言之", "最后"]):
                conclusion_text = para + "\n\n"
                break
        
        # 如果没找到明显的总结段落，生成一个基本总结
        if not conclusion_text:
            conclusion_text = f"本课程介绍了{title}的相关内容，涵盖了" + \
                              "、".join([title for title, _ in sections[:3]]) + \
                              "等方面的知识。通过学习这些内容，可以更好地理解相关概念和应用。\n\n"
        
        # 填充模板
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        notes_content = template.format(
            title=title,
            summary=summary,
            content=content,
            key_points=key_points_text,
            conclusion=conclusion_text,
            date=now
        )
        
        # 保存笔记
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(notes_content)
        
        logger.info(f"详细笔记已保存到：{output_path}")
        
        return notes_content


def create_note_generator(config: Dict[str, Any]) -> NoteGenerator:
    """
    创建笔记生成器
    
    Args:
        config: 配置字典
        
    Returns:
        笔记生成器实例
    """
    provider = config.get('note_generator', {}).get('provider', 'qwen')
    
    if provider.lower() == 'qwen':
        # 创建千问生成器
        qwen_config = config.get('note_generator', {}).get('qwen', {})
        api_key = qwen_config.get('api_key', '')
        base_url = qwen_config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        model = qwen_config.get('model', 'qwen-plus')
        system_prompt = qwen_config.get('system_prompt', None)
        
        generator = QwenNoteGenerator(api_key, base_url, model, system_prompt)
        
    else:
        raise ValueError(f"不支持的笔记生成器类型: {provider}")
    
    # 初始化生成器
    success = generator.initialize()
    if not success:
        logger.warning(f"笔记生成器 ({provider}) 初始化失败")
    
    return generator


if __name__ == "__main__":
    import argparse
    from utils import load_config
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='笔记生成器测试')
    parser.add_argument('--transcript', required=True, help='转录文本文件路径')
    parser.add_argument('--frames_dir', required=True, help='帧目录路径')
    parser.add_argument('--output', required=True, help='输出笔记文件路径')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--title', help='笔记标题')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 读取转录文本
    with open(args.transcript, 'r', encoding='utf-8') as f:
        transcript = f.read()
    
    # 获取帧信息
    frames_info = []
    for i, filename in enumerate(sorted(os.listdir(args.frames_dir))):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(args.frames_dir, filename)
            frames_info.append({
                'id': i+1,
                'file_path': file_path,
                'relative_path': filename,
                'timestamp': f"{i//60:02d}:{i%60:02d}",
                'seconds': i,
                'is_scene_change': False,
                'description': f"帧 {i+1}"
            })
    
    # 创建生成器
    generator = create_note_generator(config)
    
    # 生成笔记
    notes_content = generator.generate_notes(
        transcript,
        frames_info,
        args.output,
        title=args.title
    )
    
    print(f"笔记已生成并保存到: {args.output}")