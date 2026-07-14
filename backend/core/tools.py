# Agent工具定义
# 给Agent提供两个工具：知识库查询 和 网络搜索

from langchain_core.tools import tool


@tool
def database_query(query: str) -> str:
    """从知识库查询信息，用于回答关于文档内容的问题"""
    from backend.core.database import DatabaseManager
    db = DatabaseManager()
    results = db.search_similar(query, top_k=3)
    if not results:
        return "知识库中没有找到相关信息"
    parts = []
    for r in results:
        source = r.get("metadata", {}).get("filename", "未知")
        parts.append(f"【来源: {source}】\n{r['content']}")
    return "\n\n".join(parts)


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
    except Exception as e:
        return f"搜索出错: {e}"


def get_tools():
    """返回可用工具列表"""
    return [database_query, web_search]
