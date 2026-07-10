"""
SmartKB 数据库管理模块

本模块是系统的数据存储核心，负责管理两种数据库：

┌─────────────────────────────────────────────────────────────────┐
│                       数据存储架构                              │
├─────────────────────────────────────────────────────────────────┤
│ SQLite（关系型数据库）        │ ChromaDB（向量数据库）          │
│ ├─ documents: 文档元信息     │ ├─ 存储文档向量                 │
│ ├─ qa_records: 问答记录      │ ├─ 语义相似度检索               │
│ ├─ feedback: 用户反馈        │ └─ 支持批量增删改查             │
│ └─ conversations: 对话历史   │                                 │
└─────────────────────────────────────────────────────────────────┘

设计原则：
1. SQLite 用于事务性数据，保证ACID特性
2. ChromaDB 用于向量检索，支持语义搜索
3. 通过 DatabaseManager 统一管理，屏蔽底层实现细节

使用示例：
    db = DatabaseManager(sqlite_path="data/smartkb.db", chroma_path="data/chroma_db")
    db.add_document(doc_info)           # 添加文档
    results = db.search_similar("查询")  # 语义检索
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime

import chromadb

from backend.models.schemas import DocumentInfo, DocumentChunk, DocumentStatus


# ============================================================================
# 数据库上下文管理器
# ============================================================================

@contextmanager
def _sqlite_connection(db_path: str):
    """SQLite 连接的上下文管理器，确保异常时也关闭连接"""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


# ============================================================================
# 数据库表结构定义
# ============================================================================

# 文档表DDL
DOCUMENTS_TABLE = '''
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,                    -- 文档唯一ID (UUID)
        filename TEXT NOT NULL,                 -- 原始文件名
        upload_time TIMESTAMP,                  -- 上传时间
        status TEXT,                            -- 解析状态: pending/processing/success/failed
        chunk_count INTEGER DEFAULT 0,          -- 分块数量
        file_size INTEGER DEFAULT 0             -- 文件大小(字节)
    )
'''

# 问答记录表DDL
QA_RECORDS_TABLE = '''
    CREATE TABLE IF NOT EXISTS qa_records (
        id TEXT PRIMARY KEY,                    -- 记录ID
        question TEXT,                          -- 用户问题
        answer TEXT,                            -- 系统回答
        response_time FLOAT,                    -- 响应时间(秒)
        conversation_id TEXT,                   -- 所属对话ID
        created_at TIMESTAMP                    -- 创建时间
    )
'''

# 反馈表DDL
FEEDBACK_TABLE = '''
    CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY,                    -- 反馈ID
        message_id TEXT,                        -- 对应的消息ID
        is_positive BOOLEAN,                    -- True=点赞, False=点踩
        created_at TIMESTAMP                    -- 创建时间
    )
'''

# 对话历史表DDL
CONVERSATIONS_TABLE = '''
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,                    -- 记录ID
        role TEXT,                              -- 角色: user/assistant
        content TEXT,                           -- 消息内容
        conversation_id TEXT,                   -- 所属对话ID
        created_at TIMESTAMP                    -- 创建时间
    )
'''


class DatabaseManager:
    """
    数据库管理器

    统一管理 SQLite 和 ChromaDB，提供：
    - 文档CRUD操作
    - 向量检索
    - 问答记录管理
    - 用户反馈统计
    - 对话历史管理

    Attributes:
        sqlite_path: SQLite数据库文件路径
        chroma_path: ChromaDB数据目录
        chroma_client: ChromaDB客户端
        collection: 文档向量集合
    """

    def __init__(self, sqlite_path: str = "data/smartkb.db", chroma_path: str = "data/chroma_db"):
        """
        初始化数据库管理器

        Args:
            sqlite_path: SQLite数据库文件路径
            chroma_path: ChromaDB数据存储目录
        """
        self.sqlite_path = sqlite_path
        self.chroma_path = chroma_path

        # 确保数据目录存在
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        os.makedirs(chroma_path, exist_ok=True)

        # 初始化SQLite表结构
        self._init_sqlite()

        # 初始化ChromaDB客户端
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # 获取或创建文档集合（使用余弦相似度）
        self.collection = self.chroma_client.get_or_create_collection(
            name="smartkb_documents",
            metadata={"hnsw:space": "cosine"}   # 使用余弦相似度算法
        )

    def _init_sqlite(self):
        """
        初始化SQLite数据库表结构

        创建4张核心表：
        - documents: 文档元信息
        - qa_records: 问答记录
        - feedback: 用户反馈
        - conversations: 对话历史
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(DOCUMENTS_TABLE)
            cursor.execute(QA_RECORDS_TABLE)
            cursor.execute(FEEDBACK_TABLE)
            cursor.execute(CONVERSATIONS_TABLE)
            conn.commit()

    # ========================================================================
    # 文档管理 (Document CRUD)
    # ========================================================================

    def add_document(self, doc_info: DocumentInfo):
        """
        添加文档记录

        将文档元信息插入SQLite数据库
        Args:
            doc_info: 文档信息对象，包含id、filename、status等字段
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO documents (id, filename, upload_time, status, chunk_count, file_size)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                doc_info.id,
                doc_info.filename,
                doc_info.upload_time.isoformat(),
                doc_info.status.value,
                doc_info.chunk_count,
                doc_info.file_size
            ))
            conn.commit()

    def update_document_status(self, doc_id: str, status: DocumentStatus, chunk_count: int = 0):
        """
        更新文档解析状态
        用于文档解析过程中的状态更新：
        pending → processing → success/failed

        Args:
            doc_id: 文档ID
            status: 新状态
            chunk_count: 分块数量（解析完成后更新）
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE documents SET status = ?, chunk_count = ? WHERE id = ?
            ''', (status.value, chunk_count, doc_id))
            conn.commit()

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        获取单个文档信息

        Args:
            doc_id: 文档ID

        Returns:
            文档信息字典，不存在返回None
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return {
                "id": row[0],
                "filename": row[1],
                "upload_time": row[2],
                "status": row[3],
                "chunk_count": row[4],
                "file_size": row[5]
            }

    def get_all_documents(self) -> List[Dict]:
        """
        获取所有文档列表
        按上传时间倒序排列（最新的在前）
        Returns:
            文档信息字典列表
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM documents ORDER BY upload_time DESC')
            rows = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "filename": row[1],
                    "upload_time": row[2],
                    "status": row[3],
                    "chunk_count": row[4],
                    "file_size": row[5]
                }
                for row in rows
            ]

    def delete_document(self, doc_id: str):
        """
        删除文档

        同时删除SQLite中的记录和ChromaDB中的向量数据

        Args:
            doc_id: 文档ID
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            conn.commit()

        # 2. 从ChromaDB删除向量数据
        try:
            results = self.collection.get(where={"document_id": doc_id})
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
        except Exception as e:
            print(f"[警告] 删除ChromaDB数据时出错: {e}")

    # ========================================================================
    # 向量检索 (Vector Search)
    # ========================================================================

    def add_chunks(self, chunks: List[DocumentChunk]):
        """
        批量添加文档分块到向量数据库

        Args:
            chunks: 文档分块列表
        """
        if not chunks:
            return

        # 提取分块数据
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # 批量添加到ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas        # 元数据可用于过滤
        )

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        语义检索相似文档

        原理：
        1. 将查询文本转换为向量
        2. 在ChromaDB中计算与所有文档的余弦相似度
        3. 返回最相似的top_k个结果

        Args:
            query: 查询文本
            top_k: 返回结果数量，默认3

        Returns:
            搜索结果列表，每个结果包含：
            - content: 文档内容
            - metadata: 元数据（文件名、分块索引等）
            - score: 相似度分数（越小越相似）
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        if not results or not results['documents']:
            return []

        # 整理搜索结果
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "score": results['distances'][0][i] if results['distances'] else 0
            })

        return search_results

    # ========================================================================
    # 问答记录 (QA Records)
    # ========================================================================

    def add_qa_record(self, record_id: str, question: str, answer: str,
                      response_time: float, conversation_id: str):
        """
        添加问答记录

        Args:
            record_id: 记录ID
            question: 用户问题
            answer: 系统回答
            response_time: 响应时间（秒）
            conversation_id: 所属对话ID
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO qa_records (id, question, answer, response_time, conversation_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (record_id, question, answer, response_time, conversation_id, datetime.now().isoformat()))
            conn.commit()

    def get_qa_stats(self) -> Dict:
        """
        获取问答统计数据

        返回以下统计指标：
        - total_documents: 文档总数
        - total_qa_count: 总问答次数
        - today_qa_count: 今日问答次数
        - avg_response_time: 平均响应时间
        - like_rate: 点赞率（点赞数/总反馈数）
        Returns:
            统计数据字典
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()

            # 总问答数
            cursor.execute('SELECT COUNT(*) FROM qa_records')
            total_qa = cursor.fetchone()[0]

            # 今日问答数
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('SELECT COUNT(*) FROM qa_records WHERE created_at LIKE ?', (f'{today}%',))
            today_qa = cursor.fetchone()[0]

            # 平均响应时间
            cursor.execute('SELECT AVG(response_time) FROM qa_records')
            avg_time = cursor.fetchone()[0] or 0

            # 点赞率 = 点赞数 / 总反馈数
            cursor.execute('SELECT COUNT(*) FROM feedback WHERE is_positive = 1')
            likes = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM feedback')
            total_feedback = cursor.fetchone()[0]
            like_rate = likes / total_feedback if total_feedback > 0 else 0

            return {
                "total_documents": self._get_document_count(),
                "total_qa_count": total_qa,
                "today_qa_count": today_qa,
                "avg_response_time": round(avg_time, 2),
                "like_rate": round(like_rate, 2)
            }

    def _get_document_count(self) -> int:
        """获取文档总数"""
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM documents')
            return cursor.fetchone()[0]

    # ========================================================================
    # 反馈管理 (Feedback)
    # ========================================================================

    def add_feedback(self, message_id: str, is_positive: bool):
        """
        添加用户反馈

        使用INSERT OR REPLACE实现：
        - 首次反馈：插入新记录
        - 重复反馈：更新原有记录
        Args:
            message_id: 消息ID
            is_positive: True=点赞, False=点踩
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()
            feedback_id = f"fb_{message_id}"
            cursor.execute('''
                INSERT OR REPLACE INTO feedback (id, message_id, is_positive, created_at)
                VALUES (?, ?, ?, ?)
            ''', (feedback_id, message_id, is_positive, datetime.now().isoformat()))
            conn.commit()

    # ========================================================================
    # 对话管理 (Conversation)
    # ========================================================================

    def add_conversation(self, conversation_id: str, role: str, content: str):
        """
        添加对话记录

        Args:
            conversation_id: 对话ID（用于区分不同对话会话）
            role: 角色（user=用户, assistant=AI助手）
            content: 消息内容
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()

            # 生成唯一记录ID（使用时间戳保证唯一性）
            record_id = f"conv_{conversation_id}_{datetime.now().timestamp()}"
            cursor.execute('''
                INSERT INTO conversations (id, role, content, conversation_id, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (record_id, role, content, conversation_id, datetime.now().isoformat()))
            conn.commit()

    def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict]:
        """
        获取对话历史

        返回最近limit条消息，按时间正序排列（最早的在前）
        Args:
            conversation_id: 对话ID
            limit: 返回的最大消息数，默认10

        Returns:
            对话历史列表，格式：[{"role": "user", "content": "..."}, ...]
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()

            # 查询最近的消息（按时间倒序）
            cursor.execute('''
                SELECT role, content FROM conversations
                WHERE conversation_id = ?
                ORDER BY created_at DESC LIMIT ?
            ''', (conversation_id, limit))
            rows = cursor.fetchall()

            # 反转顺序，使最早的对话在前
            return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

    def clear_conversation(self, conversation_id: str):
        """
        清空对话历史

        删除指定对话ID的所有消息记录
        Args:
            conversation_id: 对话ID
        """
        with _sqlite_connection(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM conversations WHERE conversation_id = ?', (conversation_id,))
            conn.commit()
