"""
API路由 - 定义所有接口
"""

import os
import uuid
import time
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
from backend.config import load_config

router = APIRouter()
db_manager = DatabaseManager()
config = load_config()

# 初始化各模块
parser = DocumentParser(
    chunk_size=config["app"]["chunk_size"],
    chunk_overlap=config["app"]["chunk_overlap"]
)
rag_engine = RAGEngine(db_manager, config)
agent_engine = AgentEngine(rag_engine, db_manager, config)
eval_engine = EvalEngine(rag_engine, db_manager)
alert_manager = AlertManager(config.get("alert", {}))


# ---- 文档管理 ----

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传并解析文档"""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {'.pdf', '.txt', '.md', '.markdown'}:
        raise HTTPException(400, f"不支持的文件格式: {ext}")

    doc_id = str(uuid.uuid4())
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{doc_id}{ext}")

    # 保存文件
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)

    try:
        # 解析并入库
        doc_info, chunks = parser.parse_document(file_path, doc_id)
        db_manager.add_document(doc_info)
        db_manager.add_chunks(chunks)

        return {
            "success": True,
            "document_id": doc_id,
            "filename": file.filename,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        raise HTTPException(500, f"解析失败: {e}")


@router.get("/documents")
async def list_documents():
    """获取所有文档"""
    docs = db_manager.get_all_documents()
    return {"documents": docs, "total": len(docs)}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档"""
    db_manager.delete_document(doc_id)
    return {"success": True, "message": "删除成功"}


# ---- 对话问答 ----

@router.post("/chat")
async def chat(request: dict):
    """知识库问答"""
    question = request.get("question", "")
    conv_id = request.get("conversation_id", str(uuid.uuid4()))

    if not question.strip():
        raise HTTPException(400, "问题不能为空")

    start = time.time()
    try:
        # 先尝试纯RAG
        docs = rag_engine.retrieve(question)
        if docs:
            context = rag_engine.format_docs(docs)
            answer = rag_engine.generate(question, context)
        else:
            # 知识库没有，用Agent搜索
            answer = agent_engine.chat(question, conv_id)

        elapsed = time.time() - start

        # 记录问答
        record_id = str(uuid.uuid4())
        db_manager.add_qa_record(record_id, question, answer, elapsed, conv_id)
        db_manager.add_conversation(conv_id, "user", question)
        db_manager.add_conversation(conv_id, "assistant", answer)

        # 检查是否需要告警（响应时间过长）
        if elapsed > 10:
            alert_manager.send("响应时间告警", f"问题: {question}\n耗时: {elapsed:.1f}秒", "warning")

        return {
            "answer": answer,
            "conversation_id": conv_id,
            "response_time": round(elapsed, 2),
            "sources": [{"filename": d.get("metadata", {}).get("filename", "")} for d in docs]
        }
    except Exception as e:
        raise HTTPException(500, f"问答出错: {e}")


@router.post("/chat/stream")
async def chat_stream(request: dict):
    """流式问答"""
    question = request.get("question", "")
    conv_id = request.get("conversation_id", str(uuid.uuid4()))

    if not question.strip():
        raise HTTPException(400, "问题不能为空")

    docs = rag_engine.retrieve(question)
    context = rag_engine.format_docs(docs) if docs else ""

    def generate():
        for chunk in rag_engine.generate_stream(question, context):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """获取对话历史"""
    history = db_manager.get_conversation_history(conv_id)
    return {"conversation_id": conv_id, "messages": history}


@router.delete("/conversations/{conv_id}")
async def clear_conversation(conv_id: str):
    """清空对话历史"""
    db_manager.clear_conversation(conv_id)
    return {"success": True}


# ---- 反馈 ----

@router.post("/feedback")
async def add_feedback(request: dict):
    """添加反馈"""
    message_id = request.get("message_id", "")
    is_positive = request.get("is_positive", True)
    db_manager.add_feedback(message_id, is_positive)
    return {"success": True}


# ---- 评估 ----

@router.post("/eval")
async def run_eval(request: dict):
    """运行评估"""
    qa_pairs = request.get("qa_pairs", [])
    if not qa_pairs:
        raise HTTPException(400, "请提供评估数据")

    results = eval_engine.evaluate_qa_pairs(qa_pairs)
    summary = eval_engine.get_metrics_summary(results)
    return {"results": results, "summary": summary}


# ---- 统计 ----

@router.get("/stats")
async def get_stats():
    """获取系统统计"""
    return db_manager.get_qa_stats()


# ---- 健康检查 ----

@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
