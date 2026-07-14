"""
API路由 - 14个RESTful端点
覆盖：文档管理(3) + 智能问答(4) + 反馈(1) + RAG评估(2) + 配置管理(2) + 系统监控(2)
"""

import os
import uuid
import time
import json
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.core.database import DatabaseManager
from backend.core.parser import DocumentParser
from backend.core.rag_engine import RAGEngine
from backend.core.agent_engine import AgentEngine
from backend.core.eval_engine import EvalEngine
from backend.core.alert import AlertManager
from backend.config import load_config, reload_config, save_config

router = APIRouter()
db_manager = DatabaseManager()
config = load_config()

parser = DocumentParser(
    chunk_size=config["app"]["chunk_size"],
    chunk_overlap=config["app"]["chunk_overlap"]
)
rag_engine = RAGEngine(db_manager, config)
agent_engine = AgentEngine(rag_engine, db_manager, config)
eval_engine = EvalEngine(rag_engine, db_manager)
alert_manager = AlertManager(config.get("alert", {}))


# ========== 文档管理（3个端点） ==========

# 1. POST /documents/upload - 上传文档
# 2. GET /documents - 文档列表
# 3. DELETE /documents/{doc_id} - 删除文档

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传并解析文档"""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {'.pdf', '.txt', '.md', '.markdown'}:
        raise HTTPException(400, f"不支持的文件格式: {ext}，支持: pdf/txt/md")

    doc_id = str(uuid.uuid4())
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{doc_id}{ext}")

    content = await file.read()

    # 检查文件大小限制（10MB）
    max_size = config.get("app", {}).get("max_file_size_mb", 10) * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(400, f"文件过大，最大支持 {config.get('app', {}).get('max_file_size_mb', 10)}MB")

    with open(file_path, 'wb') as f:
        f.write(content)

    try:
        doc_info, chunks = parser.parse_document(file_path, doc_id)
        db_manager.add_document(doc_info)
        db_manager.add_chunks(chunks)

        return {
            "success": True,
            "document_id": doc_id,
            "filename": file.filename,
            "chunk_count": len(chunks),
            "file_size": len(content),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(500, f"文档解析失败: {e}")


@router.get("/documents")
async def list_documents():
    """获取所有文档列表"""
    docs = db_manager.get_all_documents()
    return {"documents": docs, "total": len(docs)}


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """获取单个文档详情"""
    doc = db_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    return doc


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档及其向量数据"""
    doc = db_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    db_manager.delete_document(doc_id)
    return {"success": True, "message": f"文档 {doc.get('filename', doc_id)} 已删除"}


# ========== 智能问答（4个端点） ==========

# 4. POST /chat - 普通问答
# 5. POST /chat/stream - 流式问答（SSE）
# 6. GET /conversations/{conv_id} - 对话历史
# 7. DELETE /conversations/{conv_id} - 清空对话

@router.post("/chat")
async def chat(request: dict):
    """知识库问答（普通模式）"""
    question = request.get("question", "")
    conv_id = request.get("conversation_id", str(uuid.uuid4()))
    use_agent = request.get("use_agent", False)

    if not question.strip():
        raise HTTPException(400, "问题不能为空")

    start = time.time()
    try:
        # 先尝试纯RAG检索
        docs = rag_engine.retrieve(question)
        if docs and not use_agent:
            context = rag_engine.format_docs(docs)
            answer = rag_engine.generate(question, context)
            sources = rag_engine.format_sources(docs)
            mode = "rag"
        else:
            # 知识库无结果或强制Agent模式
            answer = agent_engine.chat(question, conv_id)
            sources = []
            mode = "agent"

        elapsed = time.time() - start

        # 记录问答
        record_id = str(uuid.uuid4())
        db_manager.add_qa_record(record_id, question, answer, elapsed, conv_id)
        db_manager.add_conversation(conv_id, "user", question)
        db_manager.add_conversation(conv_id, "assistant", answer)

        # 响应超时告警（异步，不阻塞响应）
        if elapsed > 10:
            alert_manager.send_async("响应时间告警", f"问题: {question}\n耗时: {elapsed:.1f}秒", "warning")

        return {
            "answer": answer,
            "conversation_id": conv_id,
            "response_time": round(elapsed, 2),
            "mode": mode,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(500, f"问答出错: {e}")


@router.post("/chat/stream")
async def chat_stream(request: dict):
    """流式问答（SSE）- 支持Agent思考过程可视化"""
    question = request.get("question", "")
    conv_id = request.get("conversation_id", str(uuid.uuid4()))

    if not question.strip():
        raise HTTPException(400, "问题不能为空")

    def event_generator():
        for event in agent_engine.chat_stream(question, conv_id):
            yield event

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """获取对话历史"""
    history = db_manager.get_conversation_history(conv_id)
    return {"conversation_id": conv_id, "messages": history, "total": len(history)}


@router.delete("/conversations/{conv_id}")
async def clear_conversation(conv_id: str):
    """清空对话历史"""
    db_manager.clear_conversation(conv_id)
    return {"success": True, "message": "对话历史已清空"}


# ========== 反馈（1个端点） ==========

# 8. POST /feedback - 提交反馈

@router.post("/feedback")
async def add_feedback(request: dict):
    """用户反馈（点赞/点踩）"""
    message_id = request.get("message_id", "")
    is_positive = request.get("is_positive", True)
    if not message_id:
        raise HTTPException(400, "message_id不能为空")
    db_manager.add_feedback(message_id, is_positive)
    return {"success": True}


# ========== RAG评估（2个端点） ==========

# 9. POST /eval/run - 运行评估
# 10. GET /eval/metrics - 获取评估指标

@router.post("/eval/run")
async def run_eval(request: dict):
    """运行RAG评估（LLM-as-a-Judge + Jaccard）"""
    qa_pairs = request.get("qa_pairs", [])
    if not qa_pairs:
        raise HTTPException(400, "请提供评估数据 qa_pairs")

    use_llm_judge = request.get("use_llm_judge", True)
    results = eval_engine.evaluate_qa_pairs(qa_pairs, use_llm_judge)
    summary = eval_engine.get_metrics_summary(results)
    return {"results": results, "summary": summary}


@router.get("/eval/metrics")
async def get_eval_metrics():
    """获取历史评估指标"""
    stats = db_manager.get_qa_stats()
    return {
        "qa_stats": stats,
        "timestamp": datetime.now().isoformat()
    }


# ========== 配置管理（2个端点） ==========

# 11. GET /config - 获取当前配置
# 12. PUT /config - 更新配置（热更新）

@router.get("/config")
async def get_config():
    """获取当前配置（隐藏敏感信息）"""
    current = load_config()
    safe_config = {
        "mode": current["app"]["mode"],
        "chunk_size": current["app"]["chunk_size"],
        "chunk_overlap": current["app"]["chunk_overlap"],
        "top_k": current["app"]["top_k"],
        "local_model": {
            "model": current["local_model"]["model"],
            "base_url": current["local_model"]["base_url"],
        },
        "cloud_model": {
            "model": current["cloud_model"]["model"],
            "api_key_set": bool(current["cloud_model"].get("api_key")),
        },
        "max_tool_calls": current.get("agent", {}).get("max_tool_calls", 5),
    }
    return safe_config


@router.put("/config")
async def update_config(request: dict):
    """热更新配置（无需重启服务）"""
    try:
        # 保存到文件
        new_config = load_config()
        for key in ["chunk_size", "chunk_overlap", "top_k"]:
            if key in request:
                new_config["app"][key] = request[key]
        if "max_tool_calls" in request:
            new_config.setdefault("agent", {})["max_tool_calls"] = request["max_tool_calls"]
        if "mode" in request:
            new_config["app"]["mode"] = request["mode"]

        save_config(new_config)

        # 热更新各引擎
        fresh_config = reload_config()
        rag_engine.reload_config(fresh_config)
        agent_engine.reload_config(fresh_config)

        return {"success": True, "message": "配置已热更新", "config": get_config.__wrapped__()}
    except Exception as e:
        raise HTTPException(500, f"配置更新失败: {e}")


# ========== 系统监控（2个端点） ==========

# 13. GET /stats - 系统统计
# 14. GET /health - 健康检查

@router.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    return db_manager.get_qa_stats()


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "mode": config["app"]["mode"],
    }
