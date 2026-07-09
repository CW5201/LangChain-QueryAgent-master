"""
SmartKB 文档解析器模块

本模块负责将上传的文档转换为可检索的文本分块。

处理流程：
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  上传文件    │ →  │  提取文本    │ →  │  智能分块    │ →  │  生成元数据  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

支持的文件格式：
┌──────────────┬───────────────────────────────────────┐
│     格式     │              说明                       │
├──────────────┼───────────────────────────────────────┤
│     PDF      │ 使用PyMuPDF提取，支持加密/损坏检测     │
│     TXT      │ 自动识别编码（UTF-8/GBK/GB2312）       │
│     Markdown │ 保留原始格式，自动识别编码              │
└──────────────┴───────────────────────────────────────┘

分块策略：
- 使用 RecursiveCharacterTextSplitter 进行语义分块
- 优先在句子边界处分割，保持语义完整性
- 分块大小和重叠度可配置

技术栈：
- PyMuPDF (fitz): PDF解析
- LangChain: 文本分块
"""

import os
import uuid
from typing import List, Tuple, Optional
from datetime import datetime

import fitz                                            # PyMuPDF库
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.models.schemas import DocumentStatus, DocumentInfo, DocumentChunk


# ============================================================================
# 常量定义
# ============================================================================

# 支持的文件格式
SUPPORTED_FORMATS = {'.pdf', '.txt', '.md', '.markdown'}

# 编码尝试列表（按优先级排序）
ENCODINGS_TO_TRY = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'utf-16']

# 文本分块分隔符（按优先级从大到小）
TEXT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]


class DocumentParser:
    """
    文档解析器
    
    将上传的文档解析为可检索的文本分块。
    
    使用示例：
        parser = DocumentParser(chunk_size=500, chunk_overlap=100)
        doc_info, chunks = parser.parse_document("path/to/document.pdf")
    
    Attributes:
        chunk_size: 分块大小（字符数），默认500
        chunk_overlap: 分块重叠（字符数），默认100
        text_splitter: LangChain文本分割器
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        初始化文档解析器
        
        Args:
            chunk_size: 分块大小，默认500字符
                - 较大值：包含更多上下文，但可能降低检索精度
                - 较小值：检索更精确，但可能丢失上下文
            chunk_overlap: 分块重叠，默认100字符
                - 确保分块边界处的语义连续性
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 创建文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=TEXT_SEPARATORS
        )
    
    # ========================================================================
    # 文本提取方法
    # ========================================================================
    
    def extract_text(self, file_path: str) -> str:
        """
        从文件中提取纯文本内容
        
        根据文件扩展名自动选择解析方法
        
        Args:
            file_path: 文件完整路径
            
        Returns:
            提取的纯文本内容
            
        Raises:
            ValueError: 不支持的文件格式
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif ext == '.txt':
            return self._extract_from_txt(file_path)
        elif ext in ['.md', '.markdown']:
            return self._extract_from_markdown(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {', '.join(SUPPORTED_FORMATS)}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """
        从PDF文件提取文本
        
        功能特性：
        - 处理加密/损坏的PDF
        - 检测纯图片页面（无法提取文本）
        - 记录解析统计信息
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            提取的文本内容，页面之间用双换行分隔
        """
        # 打开PDF文件
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise ValueError(f"无法打开PDF文件（可能已损坏或加密）: {str(e)}")
        
        text_parts = []
        stats = {
            "total_pages": len(doc),
            "pages_with_text": 0,
            "pages_with_images_only": 0,
            "empty_pages": 0
        }
        
        # 逐页提取文本
        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text()
                
                if page_text.strip():
                    # 页面有文本内容
                    text_parts.append(page_text)
                    stats["pages_with_text"] += 1
                else:
                    # 检查是否为纯图片页面
                    images = page.get_images()
                    if images:
                        stats["pages_with_images_only"] += 1
                        print(f"[解析警告] 第{page_num + 1}页为纯图片，无文本")
                    else:
                        stats["empty_pages"] += 1
            except Exception as e:
                print(f"[解析警告] 第{page_num + 1}页解析失败: {str(e)}")
                continue
        
        doc.close()
        
        # 输出统计信息
        print(f"[解析统计] 总页数: {stats['total_pages']}, "
              f"有文本: {stats['pages_with_text']}, "
              f"纯图片: {stats['pages_with_images_only']}, "
              f"空白: {stats['empty_pages']}")
        
        if not text_parts:
            raise ValueError("PDF无可提取文本（可能为纯图片扫描件，需要OCR）")
        
        return "\n\n".join(text_parts)
    
    def _extract_from_txt(self, file_path: str) -> str:
        """
        从TXT文件提取文本
        
        自动尝试多种编码，确保兼容性
        
        Args:
            file_path: TXT文件路径
            
        Returns:
            文件的完整文本内容
        """
        # 尝试各种编码
        for encoding in ENCODINGS_TO_TRY:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    print(f"[解析] 使用编码 {encoding} 成功读取")
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # 所有编码失败，使用容错模式
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                content = raw_data.decode('utf-8', errors='ignore')
                print("[解析警告] 使用容错模式，部分字符可能丢失")
                return content
        except Exception as e:
            raise ValueError(f"无法识别文件编码: {str(e)}")
    
    def _extract_from_markdown(self, file_path: str) -> str:
        """
        从Markdown文件提取文本
        
        保留原始Markdown格式
        
        Args:
            file_path: Markdown文件路径
            
        Returns:
            Markdown文件的完整内容
        """
        # 尝试各种编码
        for encoding in ENCODINGS_TO_TRY[:4]:  # 不需要utf-16
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    print(f"[解析] 使用编码 {encoding} 成功读取Markdown")
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # 容错模式
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    
    # ========================================================================
    # 文本分块方法
    # ========================================================================
    
    def split_text(self, text: str) -> List[str]:
        """
        将长文本分割成多个小块
        
        使用RecursiveCharacterTextSplitter进行智能分块：
        - 优先在段落边界(\n\n)处分割
        - 其次在句子边界(。！？)处分割
        - 最后在空格处分割
        
        Args:
            text: 待分割的文本
            
        Returns:
            文本分块列表
        """
        if not text.strip():
            return []
        
        return self.text_splitter.split_text(text)
    
    # ========================================================================
    # 完整解析流程
    # ========================================================================
    
    def parse_document(self, file_path: str, doc_id: Optional[str] = None) -> Tuple[DocumentInfo, List[DocumentChunk]]:
        """
        完整的文档解析流程
        
        ┌─────────────┐
        │  生成文档ID  │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │  提取文本    │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │  智能分块    │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │  创建分块对象 │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │  返回结果    │
        └─────────────┘
        
        Args:
            file_path: 文件路径
            doc_id: 文档ID（可选，不提供则自动生成）
            
        Returns:
            (DocumentInfo, List[DocumentChunk]) 元组
        """
        # 生成文档ID
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # 获取文件基本信息
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        try:
            # 步骤1: 提取文本
            text = self.extract_text(file_path)
            if not text.strip():
                raise ValueError("文档内容为空")
            
            # 步骤2: 智能分块
            chunks_text = self.split_text(text)
            if not chunks_text:
                raise ValueError("文本分块失败")
            
            # 步骤3: 创建分块对象列表
            chunks = self._create_chunks(doc_id, filename, chunks_text)
            
            # 步骤4: 创建文档信息
            doc_info = DocumentInfo(
                id=doc_id,
                filename=filename,
                upload_time=datetime.now(),
                status=DocumentStatus.SUCCESS,
                chunk_count=len(chunks),
                file_size=file_size
            )
            
            return doc_info, chunks
            
        except Exception as e:
            # 解析失败，返回失败状态
            doc_info = DocumentInfo(
                id=doc_id,
                filename=filename,
                upload_time=datetime.now(),
                status=DocumentStatus.FAILED,
                chunk_count=0,
                file_size=file_size
            )
            raise Exception(f"文档解析失败: {str(e)}")
    
    def _create_chunks(self, doc_id: str, filename: str, chunks_text: List[str]) -> List[DocumentChunk]:
        """
        创建分块对象列表
        
        Args:
            doc_id: 文档ID
            filename: 文件名
            chunks_text: 分块文本列表
            
        Returns:
            DocumentChunk对象列表
        """
        chunks = []
        
        for i, chunk_text in enumerate(chunks_text):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=doc_id,
                content=chunk_text,
                metadata={
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks_text)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def get_preview(self, chunks: List[DocumentChunk], num_chunks: int = 3) -> List[str]:
        """
        获取文档预览
        
        返回前N个分块的内容，用于前端预览
        
        Args:
            chunks: 分块列表
            num_chunks: 预览的分块数量，默认3
            
        Returns:
            预览文本列表
        """
        return [chunk.content for chunk in chunks[:num_chunks]]
