"""
RAG 引擎 - 负责检索知识库并生成回答
"""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.core.database import DatabaseManager
from backend.core.models_factory import create_llm, create_embeddings

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据提供的上下文回答问题。
规则：
1. 仅基于上下文信息回答
2. 没有相关信息就说"无法回答"
3. 回答要准确简洁"""


class RAGEngine:
    def __init__(self, db_manager: DatabaseManager, config: Dict[str, Any]):
        self.db = db_manager
        self.config = config
        self.llm = create_llm(config)
        self.embeddings = create_embeddings(config)

    def reload_config(self, config):
        self.config = config
        self.llm = create_llm(config)
        self.embeddings = create_embeddings(config)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """语义检索，返回最相关的文档"""
        return self.db.search_similar(query, top_k)

    def format_docs(self, docs: List[Dict]) -> str:
        """把检索结果拼成一段上下文文本"""
        parts = []
        for d in docs:
            source = d.get("metadata", {}).get("filename", "未知")
            parts.append(f"[来源: {source}]\n{d.get('content', '')}")
        return "\n\n".join(parts)

    def generate(self, question: str, context: str, history: List[Dict] = None) -> str:
        """根据上下文生成回答"""
        prompt = self._build_prompt(history)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def generate_stream(self, question: str, context: str, history: List[Dict] = None):
        """流式生成回答"""
        prompt = self._build_prompt(history)
        chain = prompt | self.llm | StrOutputParser()
        return chain.stream({"context": context, "question": question})

    def _build_prompt(self, history=None):
        """构建提示词，如果有对话历史会追加上去"""
        prompt_text = SYSTEM_PROMPT + "\n\n上下文信息：\n{context}"

        if history:
            history_text = "\n".join([
                f"{'用户' if h['role'] == 'user' else '助手'}: {h['content']}"
                for h in history[-5:]
            ])
            prompt_text = SYSTEM_PROMPT + f"\n\n对话历史：\n{history_text}\n\n上下文信息：\n{{context}}"

        return ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("human", "{question}")
        ])
