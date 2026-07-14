# 文档解析器
# 负责把上传的文件（PDF/TXT/MD）提取文本，然后切成小块

import os
import uuid
from typing import List, Tuple
from datetime import datetime

import fitz  # PyMuPDF，用来读PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.models.schemas import DocumentStatus, DocumentInfo, DocumentChunk


class DocumentParser:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        # 分块器：按中文分隔符递归切分
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        )

    def extract_text(self, file_path: str) -> str:
        """根据文件类型提取纯文本"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._read_pdf(file_path)
        elif ext in [".txt", ".md", ".markdown"]:
            return self._read_text(file_path)
        else:
            raise ValueError(f"不支持的格式: {ext}")

    def _read_pdf(self, path):
        """读取PDF文本"""
        doc = fitz.open(path)
        parts = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                parts.append(text)
        doc.close()
        if not parts:
            raise ValueError("PDF没有可提取的文字")
        return "\n\n".join(parts)

    def _read_text(self, path):
        """读取文本文件，自动检测编码"""
        # 依次尝试常见编码
        for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        # 都失败了，用容错模式
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")

    def parse_document(self, file_path: str, doc_id: str = None) -> Tuple[DocumentInfo, List[DocumentChunk]]:
        """完整的解析流程：提取文本 → 切块 → 返回文档信息和分块"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        # 提取文本
        text = self.extract_text(file_path)
        if not text.strip():
            raise ValueError("文档内容为空")

        # 切块
        chunks_text = self.splitter.split_text(text)
        if not chunks_text:
            raise ValueError("分块失败")

        # 把每个文本块封装成DocumentChunk对象
        chunks = []
        for i, ct in enumerate(chunks_text):
            chunks.append(DocumentChunk(
                id=f"{doc_id}_chunk_{i}",
                document_id=doc_id,
                content=ct,
                metadata={"document_id": doc_id, "filename": filename,
                          "chunk_index": i, "total_chunks": len(chunks_text)},
            ))

        # 文档信息
        doc_info = DocumentInfo(
            id=doc_id, filename=filename, upload_time=datetime.now(),
            status=DocumentStatus.SUCCESS, chunk_count=len(chunks), file_size=file_size,
        )
        return doc_info, chunks
