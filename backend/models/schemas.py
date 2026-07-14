"""
数据模型定义
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_time: datetime = Field(default_factory=datetime.now)
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    file_size: int = 0


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    use_agent: bool = False


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    response_time: float
    sources: List[Dict[str, Any]] = []


class FeedbackRequest(BaseModel):
    message_id: str
    is_positive: bool


class EvalRequest(BaseModel):
    qa_pairs: List[Dict[str, str]]


class EvalResult(BaseModel):
    question: str
    answer: str
    expected_answer: str
    context: str
    response_time: float
    metrics: Dict[str, Any]


class SystemStats(BaseModel):
    total_documents: int = 0
    total_qa_count: int = 0
    today_qa_count: int = 0
    avg_response_time: float = 0.0
    like_rate: float = 0.0
