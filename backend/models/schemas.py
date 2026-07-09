"""
SmartKB 数据模型定义模块

本模块定义了所有API接口使用的数据模型（Pydantic Schema）。

为什么使用Pydantic？
1. 自动数据验证 - 确保请求/响应数据格式正确
2. 自动序列化 - JSON ↔ Python对象 自动转换
3. 自动生成API文档 - FastAPI自动生成Swagger文档

模型分类：
┌──────────────────┬─────────────────────────────────────────┐
│      分类        │              包含模型                     │
├──────────────────┼─────────────────────────────────────────┤
│    文档相关      │ DocumentInfo, DocumentChunk              │
│    问答相关      │ ChatRequest, ChatResponse                │
│    配置相关      │ ModelConfig, ConfigUpdate                │
│    统计相关      │ SystemStats                              │
│    API响应       │ APIResponse（统一响应格式）               │
└──────────────────┴─────────────────────────────────────────┘
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ============================================================================
# 枚举类型定义
# ============================================================================

class ModelMode(str, Enum):
    """
    模型运行模式
    
    - CLOUD: 使用云端API（通义千问/DeepSeek）
    - LOCAL: 使用本地Ollama服务
    """
    CLOUD = "cloud"
    LOCAL = "local"


class DocumentStatus(str, Enum):
    """
    文档解析状态
    
    状态流转：PENDING → PROCESSING → SUCCESS/FAILED
    """
    PENDING = "pending"         # 等待处理
    PROCESSING = "processing"   # 处理中
    SUCCESS = "success"         # 解析成功
    FAILED = "failed"           # 解析失败


# ============================================================================
# 文档相关模型
# ============================================================================

class DocumentUpload(BaseModel):
    """
    文档上传请求
    
    用于前端上传文档时传递基本信息
    """
    filename: str           # 文件名，如"report.pdf"
    file_size: int          # 文件大小（字节）
    content_type: str       # MIME类型，如"application/pdf"


class DocumentInfo(BaseModel):
    """
    文档信息
    
    存储文档的元数据，用于文档管理和展示
    """
    id: str                           # 文档唯一ID (UUID)
    filename: str                     # 原始文件名
    upload_time: datetime             # 上传时间
    status: DocumentStatus            # 解析状态
    chunk_count: int = 0              # 分块数量（解析完成后更新）
    file_size: int = 0                # 文件大小（字节）


class DocumentChunk(BaseModel):
    """
    文档分块
    
    文档被分割成多个小块，用于向量化存储和检索
    """
    id: str                          # 分块唯一ID
    document_id: str                 # 所属文档ID
    content: str                     # 分块文本内容
    metadata: dict = {}              # 元数据（文件名、分块索引等）


# ============================================================================
# 问答相关模型
# ============================================================================

class ChatRequest(BaseModel):
    """
    问答请求
    
    前端发送问答请求时的数据格式
    """
    question: str                              # 用户问题
    conversation_id: Optional[str] = None      # 对话ID（为空则新建对话）
    use_agent: bool = True                     # 是否使用Agent模式（True=工具调用，False=纯RAG）


class ChatResponse(BaseModel):
    """
    问答响应
    
    后端返回问答结果时的数据格式
    """
    answer: str                          # AI生成的回答
    sources: List[dict] = []             # 引用来源列表
    conversation_id: str                 # 对话ID（用于多轮对话）
    thinking_process: List[str] = []     # 思考过程（展示给用户看）


class FeedbackRequest(BaseModel):
    """
    反馈请求
    
    用户对回答进行点赞/点踩时的数据格式
    """
    message_id: str       # 消息ID
    is_positive: bool     # True=点赞，False=点踩


# ============================================================================
# Agent相关模型
# ============================================================================

class AgentStep(BaseModel):
    """
    Agent执行步骤
    
    记录Agent处理过程中的每一步操作
    """
    step_type: str              # 步骤类型: thinking/tool_call/tool_result
    content: str                # 步骤内容
    tool_name: Optional[str] = None  # 工具名称（如果是工具调用）


class ToolCall(BaseModel):
    """
    工具调用记录
    
    记录Agent调用工具的详细信息
    """
    tool_name: str              # 工具名称（如database_query）
    arguments: dict             # 调用参数
    result: Optional[str] = None  # 返回结果


# ============================================================================
# 配置相关模型
# ============================================================================

class ModelConfig(BaseModel):
    """
    模型配置
    
    包含所有可配置的模型参数
    """
    mode: ModelMode = ModelMode.LOCAL                    # 运行模式
    cloud_provider: str = "dashscope"                    # 云服务商
    cloud_api_key: str = ""                              # 云端API Key
    cloud_model_name: str = "qwen-turbo"                 # 云端模型名
    local_base_url: str = "http://localhost:11434"       # Ollama服务地址
    local_model_name: str = "qwen2.5:7b"                 # 本地模型名
    temperature: float = 0.7                             # 温度（0-1，越高越随机）
    top_p: float = 0.9                                   # Top-P采样
    max_tokens: int = 2048                               # 最大生成Token数
    top_k: int = 3                                       # 检索返回的文档数量


class ConfigUpdate(BaseModel):
    """
    配置更新请求
    
    所有字段都是可选的，只更新传入的字段
    """
    mode: Optional[ModelMode] = None
    cloud_provider: Optional[str] = None
    cloud_api_key: Optional[str] = None
    cloud_model_name: Optional[str] = None
    local_base_url: Optional[str] = None
    local_model_name: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None


# ============================================================================
# 统计相关模型
# ============================================================================

class SystemStats(BaseModel):
    """
    系统统计数据
    
    用于统计面板展示
    """
    total_documents: int = 0        # 总文档数
    total_qa_count: int = 0         # 总问答次数
    today_qa_count: int = 0         # 今日问答次数
    avg_response_time: float = 0.0  # 平均响应时间（秒）
    like_rate: float = 0.0          # 点赞率（0-1）


# ============================================================================
# API响应模型
# ============================================================================

class APIResponse(BaseModel):
    """
    统一API响应格式
    
    所有接口都返回此格式，保证前端处理的一致性
    
    响应格式：
    {
        "code": 200,        // 状态码
        "msg": "success",   // 状态消息
        "data": {...}       // 响应数据（可选）
    }
    
    常用状态码：
    - 200: 成功
    - 400: 请求参数错误
    - 404: 资源不存在
    - 500: 服务器内部错误
    """
    code: int = 200                # 状态码
    msg: str = "success"           # 状态消息
    data: Optional[dict] = None    # 响应数据
