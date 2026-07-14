"""
文档解析器 - 把上传的文件变成文本分块
"""

import os
import uuid
from typing import List, Tuple, Optional
from datetime import datetime

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.models.schemas import DocumentStatus, DocumentInfo, DocumentChunk

# 分块时的分隔符，按优先级排列
SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
# 支持的文件类型
SUPPORTED = {'.pdf', '.txt', '.md', '.markdown'}
# 编码尝试顺序
ENCODINGS = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'utf-16']


class DocumentParser:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=len, separators=SEPARATORS
        )

    def extract_text(self, file_path: str) -> str:
        """根据文件类型提取纯文本"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self._read_pdf(file_path)
        elif ext == '.txt':
            return self._read_txt(file_path)
        elif ext in ['.md', '.markdown']:
            return self._read_txt(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _read_pdf(self, path):
        try:
            doc = fitz.open(path)
        except Exception as e:
            raise ValueError(f"无法打开PDF: {e}")
        parts = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                parts.append(text)
            elif page.get_images():
                print(f"[警告] 第{i+1}页是纯图片，跳过")
        doc.close()
        if not parts:
            raise ValueError("PDF没有可提取的文字")
        return "\n\n".join(parts)

    def _read_txt(self, path):
        for enc in ENCODINGS:
            try:
                with open(path, 'r', encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        # 都失败了，用容错模式
        with open(path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')

    def split_text(self, text: str) -> List[str]:
        """长文本切成小块"""
        if not text.strip():
            return []
        return self.splitter.split_text(text)

    def parse_document(self, file_path: str, doc_id: str = None) -> Tuple[DocumentInfo, List[DocumentChunk]]:
        """完整的解析流程：提取文本 -> 切块 -> 生成分块对象"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        try:
            text = self.extract_text(file_path)
            if not text.strip():
                raise ValueError("文档内容为空")

            chunks_text = self.split_text(text)
            if not chunks_text:
                raise ValueError("分块失败")

            chunks = []
            for i, ct in enumerate(chunks_text):
                chunks.append(DocumentChunk(
                    id=f"{doc_id}_chunk_{i}",
                    document_id=doc_id,
                    content=ct,
                    metadata={"document_id": doc_id, "filename": filename,
                              "chunk_index": i, "total_chunks": len(chunks_text)}
                ))

            doc_info = DocumentInfo(
                id=doc_id, filename=filename, upload_time=datetime.now(),
                status=DocumentStatus.SUCCESS, chunk_count=len(chunks), file_size=file_size
            )
            return doc_info, chunks

        except Exception as e:
            raise Exception(f"文档解析失败: {e}")
