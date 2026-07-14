"""
RAG 引擎 - 检索增强生成核心模块
负责文档检索、上下文构建、多轮对话、流式生成
"""

from typing import List, Dict, Any, Generator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.core.database import DatabaseManager
from backend.core.models_factory import create_llm, create_embeddings

SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据提供的上下文回答问题。
规则：
1. 仅基于上下文信息回答，不要编造
2. 没有相关信息就说"知识库中暂无相关信息"
3. 回答要准确简洁
4. 如果有多个来源，在回答中注明引用来源"""


class RAGEngine:
    def __init__(self, db_manager: DatabaseManager, config: Dict[str, Any]):
        self.db = db_manager
        self.config = config
        self.llm = create_llm(config)
        self.embeddings = create_embeddings(config)
        self.top_k = config.get("app", {}).get("top_k", 3)

    def reload_config(self, config):
        """热更新配置"""
        self.config = config
        self.llm = create_llm(config)
        self.embeddings = create_embeddings(config)
        self.top_k = config.get("app", {}).get("top_k", 3)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """语义检索，返回最相关的文档片段"""
        k = top_k or self.top_k
        return self.db.search_similar(query, k)

    def format_docs(self, docs: List[Dict]) -> str:
        """把检索结果拼成上下文文本，带来源标注"""
        parts = []
        for i, d in enumerate(docs):
            source = d.get("metadata", {}).get("filename", "未知")
            chunk_idx = d.get("metadata", {}).get("chunk_index", "")
            label = f"[文档{i+1}: {source}]" if chunk_idx == "" else f"[文档{i+1}: {source} 第{chunk_idx}段]"
            parts.append(f"{label}\n{d.get('content', '')}")
        return "\n\n".join(parts)

    def format_sources(self, docs: List[Dict]) -> List[Dict]:
        """提取来源信息，用于API返回"""
        sources = []
        seen = set()
        for d in docs:
            filename = d.get("metadata", {}).get("filename", "")
            if filename and filename not in seen:
                seen.add(filename)
                sources.append({
                    "filename": filename,
                    "chunk_index": d.get("metadata", {}).get("chunk_index", 0),
                    "total_chunks": d.get("metadata", {}).get("total_chunks", 0)
                })
        return sources

    def generate(self, question: str, context: str, history: List[Dict] = None) -> str:
        """根据上下文生成回答"""
        prompt = self._build_prompt(history)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def generate_stream(self, question: str, context: str, history: List[Dict] = None) -> Generator:
        """流式生成回答"""
        prompt = self._build_prompt(history)
        chain = prompt | self.llm | StrOutputParser()
        return chain.stream({"context": context, "question": question})

    def _build_prompt(self, history: List[Dict] = None) -> ChatPromptTemplate:
        """构建提示词，动态追加最近5轮对话历史"""
        prompt_text = SYSTEM_PROMPT + "\n\n上下文信息：\n{context}"

        if history:
            recent = history[-5:]
            history_lines = []
            for h in recent:
                role = "用户" if h["role"] == "user" else "助手"
                history_lines.append(f"{role}: {h['content']}")
            history_text = "\n".join(history_lines)
            prompt_text = SYSTEM_PROMPT + f"\n\n对话历史：\n{history_text}\n\n上下文信息：\n{{context}}"

        return ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("human", "{question}")
        ])
