# 数据库管理
# SQLite存业务数据（文档、问答记录、反馈）
# ChromaDB存向量（用于语义检索）

import os
import sqlite3
from contextlib import contextmanager
from typing import List, Optional, Dict
from datetime import datetime
import chromadb

from backend.models.schemas import DocumentInfo, DocumentChunk, DocumentStatus


# 连接sqlite的工具函数，用完自动关闭
@contextmanager
def get_db(db_path):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


class DatabaseManager:
    def __init__(self, sqlite_path="data/smartkb.db", chroma_path="data/chroma_db"):
        # 确保目录存在
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        os.makedirs(chroma_path, exist_ok=True)

        self.sqlite_path = sqlite_path

        # 初始化sqlite表
        with get_db(sqlite_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, filename TEXT, upload_time TEXT,
                status TEXT, chunk_count INTEGER DEFAULT 0, file_size INTEGER DEFAULT 0
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS qa_records (
                id TEXT PRIMARY KEY, question TEXT, answer TEXT,
                response_time REAL, conversation_id TEXT, created_at TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY, message_id TEXT, is_positive INTEGER, created_at TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, role TEXT, content TEXT,
                conversation_id TEXT, created_at TEXT
            )""")
            conn.commit()

        # 初始化向量数据库
        self.chroma = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma.get_or_create_collection(
            name="smartkb_docs",
            metadata={"hnsw:space": "cosine"}  # 用余弦相似度
        )

    # ========== 文档操作 ==========

    def add_document(self, doc: DocumentInfo):
        """添加文档记录"""
        with get_db(self.sqlite_path) as conn:
            conn.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                (doc.id, doc.filename, str(doc.upload_time),
                 doc.status.value, doc.chunk_count, doc.file_size)
            )
            conn.commit()

    def update_document_status(self, doc_id, status, chunk_count=0):
        """更新文档状态"""
        with get_db(self.sqlite_path) as conn:
            conn.execute(
                "UPDATE documents SET status=?, chunk_count=? WHERE id=?",
                (status.value, chunk_count, doc_id)
            )
            conn.commit()

    def get_document(self, doc_id) -> Optional[Dict]:
        """获取单个文档"""
        with get_db(self.sqlite_path) as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                return None
            return {"id": row[0], "filename": row[1], "upload_time": row[2],
                    "status": row[3], "chunk_count": row[4], "file_size": row[5]}

    def get_all_documents(self) -> List[Dict]:
        """获取所有文档"""
        with get_db(self.sqlite_path) as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY upload_time DESC").fetchall()
            return [{"id": r[0], "filename": r[1], "upload_time": r[2],
                     "status": r[3], "chunk_count": r[4], "file_size": r[5]} for r in rows]

    def delete_document(self, doc_id):
        """删除文档（同时删除对应的向量）"""
        with get_db(self.sqlite_path) as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            conn.commit()
        # 删除向量数据
        try:
            results = self.collection.get(where={"document_id": doc_id})
            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
        except Exception:
            pass

    # ========== 向量操作 ==========

    def add_chunks(self, chunks: List[DocumentChunk]):
        """把文档分块存入向量数据库"""
        if not chunks:
            return
        self.collection.add(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def search_similar(self, query: str, top_k=3) -> List[Dict]:
        """语义检索：根据问题找到最相关的文档片段"""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        if not results or not results["documents"]:
            return []
        out = []
        for i in range(len(results["documents"][0])):
            out.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": results["distances"][0][i] if results["distances"] else 0,
            })
        return out

    # ========== 问答记录 ==========

    def add_qa_record(self, record_id, question, answer, response_time, conv_id):
        """记录一次问答"""
        with get_db(self.sqlite_path) as conn:
            conn.execute(
                "INSERT INTO qa_records VALUES (?,?,?,?,?,?)",
                (record_id, question, answer, response_time, conv_id, datetime.now().isoformat())
            )
            conn.commit()

    def get_qa_stats(self) -> Dict:
        """获取统计信息"""
        with get_db(self.sqlite_path) as conn:
            total_qa = conn.execute("SELECT COUNT(*) FROM qa_records").fetchone()[0]
            today = datetime.now().strftime("%Y-%m-%d")
            today_qa = conn.execute("SELECT COUNT(*) FROM qa_records WHERE created_at LIKE ?",
                                    (f"{today}%",)).fetchone()[0]
            avg_time = conn.execute("SELECT AVG(response_time) FROM qa_records").fetchone()[0] or 0
            likes = conn.execute("SELECT COUNT(*) FROM feedback WHERE is_positive=1").fetchone()[0]
            total_fb = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            return {
                "total_documents": doc_count,
                "total_qa_count": total_qa,
                "today_qa_count": today_qa,
                "avg_response_time": round(avg_time, 2),
                "like_rate": round(likes / total_fb, 2) if total_fb > 0 else 0,
            }

    # ========== 反馈 ==========

    def add_feedback(self, message_id, is_positive):
        """添加用户反馈（点赞/点踩）"""
        with get_db(self.sqlite_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO feedback VALUES (?,?,?,?)",
                (f"fb_{message_id}", message_id, int(is_positive), datetime.now().isoformat())
            )
            conn.commit()

    # ========== 对话历史 ==========

    def add_conversation(self, conv_id, role, content):
        """保存一条对话记录"""
        with get_db(self.sqlite_path) as conn:
            rid = f"conv_{conv_id}_{datetime.now().timestamp()}"
            conn.execute(
                "INSERT INTO conversations VALUES (?,?,?,?,?)",
                (rid, role, content, conv_id, datetime.now().isoformat())
            )
            conn.commit()

    def get_conversation_history(self, conv_id, limit=10) -> List[Dict]:
        """获取对话历史（最近N条）"""
        with get_db(self.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversations WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
                (conv_id, limit)
            ).fetchall()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def clear_conversation(self, conv_id):
        """清空对话历史"""
        with get_db(self.sqlite_path) as conn:
            conn.execute("DELETE FROM conversations WHERE conversation_id=?", (conv_id,))
            conn.commit()
