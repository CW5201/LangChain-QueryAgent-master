# API路由
# 定义所有接口，共15个端点

import os
import uuid
import time
import json
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from backend.core.database import DatabaseManager
from backend.core.parser import DocumentParser
from backend.core.rag_engine import RAGEngine
from backend.core.agent_engine import AgentEngine
from backend.core.eval_engine import EvalEngine
from backend.core.alert import AlertManager
from backend.config import load_config, reload_config

router = APIRouter()

# 初始化各模块
db = DatabaseManager()
config = load_config()
parser = DocumentParser(config["app"]["chunk_size"], config["app"]["chunk_overlap"])
rag = RAGEngine(db, config)
agent = AgentEngine(rag, config)
eval_engine = EvalEngine(rag)
alert = AlertManager(config.get("alert", {}))


# ========== 文档管理 ==========

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档"""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pdf", ".txt", ".md", ".markdown"}:
        raise HTTPException(400, f"不支持的格式: {ext}")

    doc_id = str(uuid.uuid4())
    os.makedirs("data/uploads", exist_ok=True)
    path = f"data/uploads/{doc_id}{ext}"

    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)

    doc_info, chunks = parser.parse_document(path, doc_id)
    db.add_document(doc_info)
    db.add_chunks(chunks)
    return {"success": True, "document_id": doc_id, "filename": file.filename, "chunk_count": len(chunks)}


@router.get("/documents")
async def list_documents():
    """文档列表"""
    docs = db.get_all_documents()
    return {"documents": docs, "total": len(docs)}


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """文档详情"""
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    return doc


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档"""
    db.delete_document(doc_id)
    return {"success": True}


# ========== 智能问答 ==========

@router.post("/chat")
async def chat(request: dict):
    """普通问答"""
    question = request.get("question", "")
    conv_id = request.get("conversation_id", str(uuid.uuid4()))
    if not question.strip():
        raise HTTPException(400, "问题不能为空")

    start = time.time()
    docs = rag.retrieve(question)
    if docs:
        context = rag.format_docs(docs)
        answer = rag.generate(question, context)
        sources = rag.format_sources(docs)
        mode = "rag"
    else:
        answer = agent.chat(question, conv_id)
        sources = []
        mode = "agent"

    elapsed = time.time() - start
    db.add_qa_record(str(uuid.uuid4()), question, answer, elapsed, conv_id)
    db.add_conversation(conv_id, "user", question)
    db.add_conversation(conv_id, "assistant", answer)

    if elapsed > 10:
        alert.send_async("响应超时告警", f"问题: {question}\n耗时: {elapsed:.1f}秒", "warning")

    return {"answer": answer, "conversation_id": conv_id, "response_time": round(elapsed, 2), "mode": mode, "sources": sources}


@router.post("/chat/stream")
async def chat_stream(request: dict):
    """流式问答（SSE）"""
    question = request.get("question", "")
    conv_id = request.get("conversation_id", str(uuid.uuid4()))
    if not question.strip():
        raise HTTPException(400, "问题不能为空")
    return StreamingResponse(agent.chat_stream(question, conv_id), media_type="text/event-stream")


@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """对话历史"""
    history = db.get_conversation_history(conv_id)
    return {"conversation_id": conv_id, "messages": history}


@router.delete("/conversations/{conv_id}")
async def clear_conversation(conv_id: str):
    """清空对话"""
    db.clear_conversation(conv_id)
    return {"success": True}


# ========== 反馈 ==========

@router.post("/feedback")
async def add_feedback(request: dict):
    """点赞/点踩"""
    db.add_feedback(request.get("message_id", ""), request.get("is_positive", True))
    return {"success": True}


# ========== RAG评估 ==========

@router.post("/eval/run")
async def run_eval(request: dict):
    """运行评估"""
    qa_pairs = request.get("qa_pairs", [])
    if not qa_pairs:
        raise HTTPException(400, "请提供qa_pairs")
    results = eval_engine.evaluate_qa_pairs(qa_pairs, request.get("use_llm_judge", True))
    return {"results": results, "summary": eval_engine.get_metrics_summary(results)}


@router.get("/eval/metrics")
async def get_eval_metrics():
    """获取评估指标"""
    return db.get_qa_stats()


# ========== 配置管理 ==========

@router.get("/config")
async def get_config():
    """获取配置"""
    c = load_config()
    return {"mode": c["app"]["mode"], "chunk_size": c["app"]["chunk_size"],
            "chunk_overlap": c["app"]["chunk_overlap"], "top_k": c["app"]["top_k"],
            "max_tool_calls": c["agent"]["max_tool_calls"]}


@router.put("/config")
async def update_config(request: dict):
    """热更新配置（不用重启）"""
    new = load_config()
    for key in ["chunk_size", "chunk_overlap", "top_k"]:
        if key in request:
            new["app"][key] = request[key]
    if "max_tool_calls" in request:
        new["agent"]["max_tool_calls"] = request["max_tool_calls"]
    fresh = reload_config()
    rag.reload_config(fresh)
    agent.reload_config(fresh)
    return {"success": True, "message": "配置已更新"}


# ========== 系统监控 ==========

@router.get("/stats")
async def get_stats():
    """系统统计"""
    return db.get_qa_stats()


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "version": "1.0.0", "timestamp": datetime.now().isoformat()}
