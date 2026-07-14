"""
工具定义 - 给Agent提供数据库查询和网络搜索能力
"""

from typing import Optional
from langchain_core.tools import tool

from backend.core.database import DatabaseManager


@tool
def database_query(query: str) -> str:
    """从知识库中查询相关信息，用于回答关于文档内容的问题"""
    try:
        db = DatabaseManager()
        results = db.search_similar(query, top_k=3)
        if not results:
            return "知识库中没有找到相关信息"
        parts = []
        for i, r in enumerate(results):
            source = r.get("metadata", {}).get("filename", "未知")
            parts.append(f"【来源: {source}】\n{r['content']}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"查询出错: {e}"


@tool
def web_search(query: str) -> str:
    """从互联网搜索最新信息，用于回答实时性问题"""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "搜索无结果"
        parts = []
        for r in results:
            parts.append(f"【{r.get('title', '')}】\n{r.get('body', '')}")
        return "\n\n".join(parts)
    except ImportError:
        return "搜索功能需要安装ddgs包"
    except Exception as e:
        return f"搜索出错: {e}"


def get_tools(db_manager: Optional[DatabaseManager] = None):
    """返回可用的工具列表"""
    return [database_query, web_search]
