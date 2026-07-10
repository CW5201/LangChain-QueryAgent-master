"""
SmartKB API路由模块

本模块定义了所有RESTful API接口。

接口总览：
┌──────────────────────────┬──────────────────────────────────────┐
│       接口路径           │             功能说明                 │
├──────────────────────────┼──────────────────────────────────────┤
│ POST /documents/upload   │ 上传文档                             │
│ GET  /documents          │ 获取文档列表                         │
│ GET  /documents/{id}     │ 获取文档详情                         │
│ DELETE /documents/{id}   │ 删除文档                             │
│ GET  /documents/{id}/preview │ 预览文档分块                    │
├──────────────────────────┼──────────────────────────────────────┤
│ POST /chat               │ 智能问答（非流式）                   │
│ POST /chat/stream        │ 智能问答（流式SSE）                 │
│ POST /chat/feedback      │ 提交反馈                             │
│ DELETE /chat/conversation/{id} │ 清空对话历史                  │
├──────────────────────────┼──────────────────────────────────────┤
│ GET  /config             │ 获取配置                             │
│ PUT  /config             │ 更新配置（热更新）                   │
├──────────────────────────┼──────────────────────────────────────┤
│ GET  /stats              │ 获取系统统计                         │
│ GET  /health             │ 健康检查                             │
│ POST /eval               │ RAG质量评估                          │
│ GET  /eval/history       │ 获取评估历史                         │
└──────────────────────────┴──────────────────────────────────────┘

接口设计原则：
1. 统一响应格式: {"code": 200, "msg": "success", "data": {...}}
2. RESTful风格: 资源用名词，操作用动词
3. 支持CORS跨域

本模块使用FastAPI Depends 进行依赖注入，避免全局可变状态。
"""

import os
import uuid
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import StreamingResponse

from backend.models.schemas import (
    DocumentInfo, DocumentStatus, APIResponse,
    ChatRequest, FeedbackRequest, ConfigUpdate
)
from backend.core.parser import DocumentParser
from backend.core.database import DatabaseManager
from backend.core.rag_engine import RAGEngine
from backend.core.agent_engine import AgentEngine


# ============================================================================
# 路由模块
# ============================================================================

# 创建路由器
router = APIRouter()

# 上传文件存储目录
UPLOAD_DIR = "data/uploads"


# ============================================================================
# 依赖注入（由 main.py lifespan 初始化）
# ============================================================================

# 全局实例（在main.py启动时通过set_dependencies注入）
_db_manager: Optional[DatabaseManager] = None
_rag_engine: Optional[RAGEngine] = None
_agent_engine: Optional[AgentEngine] = None
_parser: Optional[DocumentParser] = None
_app_config: dict = {}


def set_dependencies(database_manager: DatabaseManager, rag: RAGEngine,
                     agent: AgentEngine, cfg: dict):
    """
    初始化路由模块依赖（替代旧版init_modules）
    在FastAPI启动时调用，注入全局实例

    Args:
        database_manager: 数据库管理器实例
        rag: RAG引擎实例
        agent: Agent引擎实例
        cfg: 系统配置字典
    """
    global _db_manager, _rag_engine, _agent_engine, _parser, _app_config

    _db_manager = database_manager
    _rag_engine = rag
    _agent_engine = agent
    _app_config = cfg

    # 初始化文档解析器
    _parser = DocumentParser(
        chunk_size=cfg.get("rag", {}).get("chunk_size", 500),
        chunk_overlap=cfg.get("rag", {}).get("chunk_overlap", 100)
    )

    # 确保上传目录存在
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db() -> DatabaseManager:
    """FastAPI Depends 依赖：获取数据库管理器"""
    if _db_manager is None:
        raise RuntimeError("DatabaseManager 未初始化，请确保已调用set_dependencies")
    return _db_manager


def get_rag() -> RAGEngine:
    """FastAPI Depends 依赖：获取RAG 引擎"""
    if _rag_engine is None:
        raise RuntimeError("RAGEngine 未初始化")
    return _rag_engine


def get_agent() -> AgentEngine:
    """FastAPI Depends 依赖：获取Agent 引擎"""
    if _agent_engine is None:
        raise RuntimeError("AgentEngine 未初始化")
    return _agent_engine


def get_cfg() -> dict:
    """FastAPI Depends 依赖：获取配置"""
    return _app_config


# ============================================================================
# 文档管理接口
# ============================================================================

@router.post("/documents/upload", response_model=APIResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: DatabaseManager = Depends(get_db),
    cfg: dict = Depends(get_cfg),
):
    """
    上传文档

    处理流程：
    ┌─────────────┐
    │ 验证文件    │ (格式、大小)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 保存文件    │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 解析文档    │ (提取文本、分块)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 向量化存储  │ (存入ChromaDB)
    └─────────────┘

    Args:
        file: 上传的文件对象
    Returns:
        包含文档ID和分块数量的APIResponse
    """
    try:
        # ---- 验证文件格式 ----
        allowed_formats = cfg.get("document", {}).get("allowed_formats", ["pdf", "txt", "md"])
        file_ext = file.filename.split(".")[-1].lower()

        if file_ext not in allowed_formats:
            return APIResponse(
                code=400,
                msg=f"不支持的格式: {file_ext}，支持 {', '.join(allowed_formats)}"
            )

        # ---- 验证文件大小 ----
        max_size_mb = cfg.get("document", {}).get("max_file_size_mb", 10)
        max_size_bytes = max_size_mb * 1024 * 1024
        content = await file.read()

        if len(content) > max_size_bytes:
            size_mb = len(content) / 1024 / 1024
            return APIResponse(code=400, msg=f"文件过大: {size_mb:.1f}MB > {max_size_mb}MB")

        # ---- 保存文件 ----
        doc_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.{file_ext}")

        with open(file_path, "wb") as f:
            f.write(content)

        # ---- 创建文档记录 ----
        doc_info = DocumentInfo(
            id=doc_id,
            filename=file.filename,
            upload_time=datetime.now(),
            status=DocumentStatus.PROCESSING,
            file_size=len(content)
        )
        db.add_document(doc_info)

        # ---- 解析文档并存储向量 ----
        try:
            doc_info, chunks = _parser.parse_document(file_path, doc_id)
            db.update_document_status(doc_id, DocumentStatus.SUCCESS, len(chunks))
            db.add_chunks(chunks)
        except Exception as e:
            db.update_document_status(doc_id, DocumentStatus.FAILED)
            return APIResponse(code=500, msg=f"文档解析失败: {str(e)}")

        return APIResponse(
            code=200,
            msg="上传成功",
            data={"id": doc_id, "filename": file.filename, "chunk_count": len(chunks)}
        )

    except Exception as e:
        return APIResponse(code=500, msg=f"上传失败: {str(e)}")


@router.get("/documents", response_model=APIResponse)
async def list_documents(db: DatabaseManager = Depends(get_db)):
    """获取所有文档列表"""
    try:
        documents = db.get_all_documents()
        return APIResponse(code=200, msg="success", data={"documents": documents})
    except Exception as e:
        return APIResponse(code=500, msg=f"获取文档列表失败: {str(e)}")


@router.get("/documents/{doc_id}", response_model=APIResponse)
async def get_document(doc_id: str, db: DatabaseManager = Depends(get_db)):
    """获取单个文档详情"""
    try:
        doc = db.get_document(doc_id)
        if not doc:
            return APIResponse(code=404, msg="文档不存在")
        return APIResponse(code=200, msg="success", data=doc)
    except Exception as e:
        return APIResponse(code=500, msg=f"获取文档详情失败: {str(e)}")


@router.delete("/documents/{doc_id}", response_model=APIResponse)
async def delete_document(doc_id: str, db: DatabaseManager = Depends(get_db)):
    """
    删除文档

    同时删除SQLite记录和ChromaDB向量数据
    """
    try:
        doc = db.get_document(doc_id)
        if not doc:
            return APIResponse(code=404, msg="文档不存在")

        db.delete_document(doc_id)

        # 删除上传的文件
        for ext in ["pdf", "txt", "md"]:
            file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.{ext}")
            if os.path.exists(file_path):
                os.remove(file_path)
                break

        return APIResponse(code=200, msg="删除成功")
    except Exception as e:
        return APIResponse(code=500, msg=f"删除失败: {str(e)}")


@router.get("/documents/{doc_id}/preview", response_model=APIResponse)
async def preview_document(doc_id: str, db: DatabaseManager = Depends(get_db)):
    """预览文档的各个分块"""
    try:
        doc = db.get_document(doc_id)
        if not doc:
            return APIResponse(code=404, msg="文档不存在")

        results = db.collection.get(where={"document_id": doc_id}, limit=3)
        chunks = results['documents'] if results and results['documents'] else []

        return APIResponse(code=200, msg="success", data={"chunks": chunks})
    except Exception as e:
        return APIResponse(code=500, msg=f"预览失败: {str(e)}")


# ============================================================================
# 智能问答接口
# ============================================================================

@router.post("/chat", response_model=APIResponse)
async def chat(
    request: ChatRequest,
    db: DatabaseManager = Depends(get_db),
    cfg: dict = Depends(get_cfg),
    agent: Optional[AgentEngine] = Depends(get_agent),
    rag: Optional[RAGEngine] = Depends(get_rag),
):
    """
    智能问答（非流式）
    支持两种模式：
    - Agent模式: 自动使用工具（数据库查询、网页搜索）
    - RAG模式: 仅检索知识库回答

    Args:
        request: 问答请求对象
    Returns:
        包含回答、思考过程、引用来源的APIResponse
    """
    try:
        # 生成对话ID
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # 获取对话历史
        max_history = cfg.get("conversation", {}).get("max_history", 10)
        conversation_history = db.get_conversation_history(conversation_id, limit=max_history)

        # 记录用户问题
        db.add_conversation(conversation_id, "user", request.question)

        # 记录开始时间
        start_time = datetime.now()

        # 处理查询
        if request.use_agent:
            result = agent.process_query(request.question, conversation_history)
        else:
            # 纯RAG模式
            search_results = rag.retrieve(request.question)
            context = rag.format_docs(search_results)
            answer = rag.generate(request.question, context, conversation_history)
            result = {
                "answer": answer,
                "thinking_process": [],
                "sources": [
                    {"filename": r["metadata"].get("filename", ""), "content": r["content"][:100]}
                    for r in search_results
                ]
            }

        # 计算响应时间
        response_time = (datetime.now() - start_time).total_seconds()

        # 记录回答
        db.add_conversation(conversation_id, "assistant", result["answer"])

        # 记录问答统计
        db.add_qa_record(
            str(uuid.uuid4()), request.question, result["answer"],
            response_time, conversation_id
        )

        return APIResponse(
            code=200,
            msg="success",
            data={
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "conversation_id": conversation_id,
                "thinking_process": result.get("thinking_process", []),
                "response_time": response_time
            }
        )

    except Exception as e:
        return APIResponse(code=500, msg=f"问答失败: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    db: DatabaseManager = Depends(get_db),
    cfg: dict = Depends(get_cfg),
    agent: AgentEngine = Depends(get_agent),
):
    """
    智能问答（流式SSE）
    使用Server-Sent Events实现打字机效果
    """
    async def event_generator():
        """SSE事件生成器"""
        conversation_id = request.conversation_id or str(uuid.uuid4())

        max_history = cfg.get("conversation", {}).get("max_history", 10)
        conversation_history = db.get_conversation_history(conversation_id, limit=max_history)

        db.add_conversation(conversation_id, "user", request.question)

        # 发送对话ID
        yield f"data: {json.dumps({'type': 'conversation_id', 'content': conversation_id}, ensure_ascii=False)}\n\n"

        full_answer = ""

        try:
            for chunk in agent.process_query_stream(request.question, conversation_history):
                yield f"data: {chunk}\n\n"

                # 安全解析JSON（防止非JSON输出导致SSE流崩溃）
                try:
                    chunk_data = json.loads(chunk)
                except (json.JSONDecodeError, TypeError):
                    continue

                if chunk_data.get("type") == "answer":
                    full_answer += chunk_data.get("content", "")

            # 记录完整回答
            if full_answer:
                db.add_conversation(conversation_id, "assistant", full_answer)

            yield f"data: {json.dumps({'type': 'done', 'content': ''}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.post("/chat/feedback", response_model=APIResponse)
async def chat_feedback(
    request: FeedbackRequest,
    db: DatabaseManager = Depends(get_db),
):
    """提交点赞/点踩反馈"""
    try:
        db.add_feedback(request.message_id, request.is_positive)
        return APIResponse(code=200, msg="反馈提交成功")
    except Exception as e:
        return APIResponse(code=500, msg=f"反馈提交失败: {str(e)}")


@router.delete("/chat/conversation/{conversation_id}", response_model=APIResponse)
async def clear_conversation(
    conversation_id: str,
    db: DatabaseManager = Depends(get_db),
):
    """清空对话历史"""
    try:
        db.clear_conversation(conversation_id)
        return APIResponse(code=200, msg="对话历史已清除")
    except Exception as e:
        return APIResponse(code=500, msg=f"清空失败: {str(e)}")


# ============================================================================
# 配置管理接口
# ============================================================================

@router.get("/config", response_model=APIResponse)
async def get_config(cfg: dict = Depends(get_cfg)):
    """获取当前配置"""
    return APIResponse(code=200, msg="success", data=cfg)


@router.put("/config", response_model=APIResponse)
async def update_config(
    update: ConfigUpdate,
    cfg: dict = Depends(get_cfg),
    rag: RAGEngine = Depends(get_rag),
    agent: AgentEngine = Depends(get_agent),
):
    """
    更新配置（热更新）
    修改后自动重新加载引擎，无需重启服务
    """
    try:
        import yaml

        # 更新配置（仅更新传入的字段）
        if update.mode is not None:
            cfg["model"]["mode"] = update.mode.value
        if update.cloud_provider is not None:
            cfg["model"]["cloud"]["provider"] = update.cloud_provider
        if update.cloud_api_key is not None:
            cfg["model"]["cloud"]["api_key"] = update.cloud_api_key
        if update.cloud_model_name is not None:
            cfg["model"]["cloud"]["model_name"] = update.cloud_model_name
        if update.local_base_url is not None:
            cfg["model"]["local"]["base_url"] = update.local_base_url
        if update.local_model_name is not None:
            cfg["model"]["local"]["model_name"] = update.local_model_name
        if update.temperature is not None:
            cfg["llm"]["temperature"] = update.temperature
        if update.top_p is not None:
            cfg["llm"]["top_p"] = update.top_p
        if update.max_tokens is not None:
            cfg["llm"]["max_tokens"] = update.max_tokens
        if update.top_k is not None:
            cfg["rag"]["top_k"] = update.top_k

        # 保存到配置文件
        with open("backend/config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        # 热更新引擎
        rag.reload_config(cfg)
        agent.reload_config(cfg)

        return APIResponse(code=200, msg="配置更新成功")

    except Exception as e:
        return APIResponse(code=500, msg=f"配置更新失败: {str(e)}")


# ============================================================================
# 统计接口
# ============================================================================

@router.get("/stats", response_model=APIResponse)
async def get_stats(db: DatabaseManager = Depends(get_db)):
    """获取系统统计数据"""
    try:
        stats = db.get_qa_stats()
        return APIResponse(code=200, msg="success", data=stats)
    except Exception as e:
        return APIResponse(code=500, msg=f"获取统计失败: {str(e)}")


# ============================================================================
# 健康检查接口
# ============================================================================

@router.get("/health")
async def health_check():
    """
    健康检查
    用于Docker和负载均衡器检测服务状态
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============================================================================
# RAG评估接口
# ============================================================================

@router.post("/eval", response_model=APIResponse)
async def eval_rag_quality(
    request: dict,
    rag: RAGEngine = Depends(get_rag),
):
    """
    RAG质量评估

    基于LLM-as-a-Judge方法评估问答质量
    """
    try:
        from backend.core.eval_engine import EvalEngine

        question = request.get("question", "")
        answer = request.get("answer", "")
        context = request.get("context", "")

        if not question or not answer:
            return APIResponse(code=400, msg="问题和回答不能为空")

        evaluator = EvalEngine(rag)
        result = evaluator.evaluate_single(
            {"question": question, "expected_keywords": []},
            top_k=3
        )

        return APIResponse(code=200, msg="评估完成", data=result)
    except Exception as e:
        return APIResponse(code=500, msg=f"评估失败: {str(e)}")


@router.get("/eval/history")
async def get_eval_history():
    """获取评估历史记录"""
    try:
        eval_file = "data/eval_results.json"
        if os.path.exists(eval_file):
            with open(eval_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            return APIResponse(code=200, msg="success", data={"records": records})
        else:
            return APIResponse(code=200, msg="暂无评估记录", data={"records": []})
    except Exception as e:
        return APIResponse(code=500, msg=f"获取评估历史失败: {str(e)}")
