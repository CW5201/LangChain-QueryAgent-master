# 数据模型定义
# 定义项目中用到的数据结构

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# 文档状态枚举
class DocumentStatus(str, Enum):
    PENDING = "pending"       # 待处理
    PROCESSING = "processing" # 处理中
    SUCCESS = "success"       # 成功
    FAILED = "failed"         # 失败


# 文档信息
class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_time: datetime = None
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    file_size: int = 0


# 文档分块
class DocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = {}
