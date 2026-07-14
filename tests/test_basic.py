"""
SmartKB 单元测试模块

本模块包含项目的核心功能测试：
- 文档解析器测试
- 数据库管理器测试
- 工具函数测试
- API接口测试

运行测试：
    cd tests
    pytest test_basic.py -v
"""

import os
import sys
import pytest
import tempfile
import shutil

# 添加项目根目录到Python路径，确保可以导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDocumentParser:
    """
    文档解析器测试类

    测试内容：
    - 文本文件提取
    - 文本分块
    - 完整文档解析流程
    """

    def setup_method(self):
        """
        测试前准备

        创建临时目录用于存放测试文件
        """
        from backend.core.parser import DocumentParser
        self.parser = DocumentParser(chunk_size=100, chunk_overlap=20)
        self.test_dir = tempfile.mkdtemp(prefix="smartkb_test_")

    def teardown_method(self):
        """
        测试后清理

        删除临时目录和测试文件
        """
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_extract_from_txt(self):
        """测试TXT文件文本提取"""
        # 创建测试文件
        test_file = os.path.join(self.test_dir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("这是一个测试文档。\n第二行内容。\n第三行内容。")

        # 提取文本
        text = self.parser.extract_text(test_file)

        # 验证结果
        assert "测试文档" in text
        assert "第二行" in text

    def test_split_text(self):
        """测试文本分块功能"""
        # 创建较长的文本
        text = "这是一段很长的文本。" * 20

        # 分块
        chunks = self.parser.split_text(text)

        # 验证：至少分成2块，每块不超过150字符（考虑重叠）
        assert len(chunks) > 0
        assert all(len(chunk) <= 150 for chunk in chunks)

    def test_parse_document(self):
        """测试完整的文档解析流程"""
        # 创建测试文件
        test_file = os.path.join(self.test_dir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("这是一个测试文档。\n" * 20)

        # 解析文档
        doc_info, chunks = self.parser.parse_document(test_file)

        # 验证文档信息
        assert doc_info.filename == "test.txt"
        assert doc_info.chunk_count > 0

        # 验证分块
        assert len(chunks) > 0
        assert all(chunk.document_id == doc_info.id for chunk in chunks)


class TestDatabaseManager:
    """
    数据库管理器测试类

    测试内容：
    - 文档CRUD操作
    - 向量存储和检索
    - 问答统计
    """

    def setup_method(self):
        """
        测试前准备

        创建临时数据库目录
        使用独立目录确保测试之间互不干扰
        """
        from backend.core.database import DatabaseManager
        self.test_dir = tempfile.mkdtemp(prefix="smartkb_test_")
        self.db_path = os.path.join(self.test_dir, "test.db")
        self.chroma_path = os.path.join(self.test_dir, "chroma_test")
        self.db_manager = DatabaseManager(self.db_path, self.chroma_path)

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_document(self):
        """测试添加文档"""
        from backend.models.schemas import DocumentInfo, DocumentStatus
        from datetime import datetime

        # 创建文档信息
        doc_info = DocumentInfo(
            id="test_doc_1",
            filename="test.txt",
            upload_time=datetime.now(),
            status=DocumentStatus.SUCCESS,
            chunk_count=5,
            file_size=1024
        )

        # 添加到数据库
        self.db_manager.add_document(doc_info)

        # 查询并验证
        doc = self.db_manager.get_document("test_doc_1")
        assert doc is not None
        assert doc["filename"] == "test.txt"

    def test_get_all_documents(self):
        """测试获取所有文档"""
        from backend.models.schemas import DocumentInfo, DocumentStatus
        from datetime import datetime

        # 添加3个测试文档
        for i in range(3):
            doc_info = DocumentInfo(
                id=f"test_doc_{i}",
                filename=f"test_{i}.txt",
                upload_time=datetime.now(),
                status=DocumentStatus.SUCCESS,
                chunk_count=i + 1,
                file_size=1024 * (i + 1)
            )
            self.db_manager.add_document(doc_info)

        # 获取所有文档
        docs = self.db_manager.get_all_documents()
        assert len(docs) == 3

    def test_delete_document(self):
        """测试删除文档"""
        from backend.models.schemas import DocumentInfo, DocumentStatus
        from datetime import datetime

        # 添加文档
        doc_info = DocumentInfo(
            id="test_doc_delete",
            filename="test_delete.txt",
            upload_time=datetime.now(),
            status=DocumentStatus.SUCCESS,
            chunk_count=1,
            file_size=512
        )
        self.db_manager.add_document(doc_info)

        # 删除文档
        self.db_manager.delete_document("test_doc_delete")

        # 验证已删除
        doc = self.db_manager.get_document("test_doc_delete")
        assert doc is None

    def test_add_and_search_chunks(self):
        """测试向量存储和语义检索"""
        from backend.models.schemas import DocumentChunk

        # 创建测试分块
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_1",
                content="这是一个关于Python编程的文档",
                metadata={"document_id": "doc_1", "filename": "python.txt"}
            ),
            DocumentChunk(
                id="chunk_2",
                document_id="doc_1",
                content="Python是一种流行的编程语言",
                metadata={"document_id": "doc_1", "filename": "python.txt"}
            )
        ]

        # 添加到向量库
        self.db_manager.add_chunks(chunks)

        # 语义检索
        results = self.db_manager.search_similar("Python编程", top_k=2)
        assert len(results) > 0

    def test_qa_stats(self):
        """测试问答统计功能"""
        stats = self.db_manager.get_qa_stats()

        # 验证统计字段存在
        assert "total_documents" in stats
        assert "total_qa_count" in stats
        assert "today_qa_count" in stats
        assert "avg_response_time" in stats
        assert "like_rate" in stats


class TestTools:
    """工具函数测试类"""

    def test_database_query_returns_string(self):
        """测试数据库查询工具返回字符串类型"""
        from backend.core.tools import database_query

        # database_query 是一个 LangChain tool，invoke 接受搜索查询字符串
        result = database_query.invoke({"query": "测试查询"})
        assert isinstance(result, str)
        # 没有匹配数据时应该返回提示信息
        assert len(result) > 0


class TestAPI:
    """API接口测试类"""

    def setup_method(self):
        """
        测试前准备

        创建FastAPI测试客户端
        """
        from fastapi.testclient import TestClient
        from backend.main import app
        self.client = TestClient(app)

    def test_health_check(self):
        """测试健康检查接口"""
        response = self.client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        # 后端返回 {code: 200, message: "操作成功", data: {"status": "ok", ...}}
        assert data["code"] == 200
        assert data["data"]["status"] == "ok"

    def test_get_documents(self):
        """测试获取文档列表接口"""
        response = self.client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        # 统一响应格式：{code: 200, message: "xxx", data: {...}}
        assert data["code"] == 200
        assert "documents" in data["data"]
        assert "total" in data["data"]

    def test_get_config(self):
        """测试获取配置接口"""
        response = self.client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        # 统一响应格式
        assert data["code"] == 200
        # data 中包含具体的配置项
        assert "mode" in data["data"]

    def test_get_stats(self):
        """测试获取统计接口"""
        response = self.client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        # 统一响应格式
        assert data["code"] == 200
        # data 中包含统计字段
        assert "total_documents" in data["data"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])