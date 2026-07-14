# RAG引擎（检索增强生成）
# 核心流程：用户提问 → 从知识库检索相关文档 → 拼成上下文 → 让大模型回答

from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.core.database import DatabaseManager
from backend.core.models_factory import create_llm, create_embeddings


# 系统提示词：告诉大模型怎么回答
SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据提供的上下文回答问题。
规则：
1. 仅基于上下文信息回答
2. 没有相关信息就说"无法回答"
3. 回答要准确简洁"""


class RAGEngine:
    def __init__(self, db_manager: DatabaseManager, config: Dict):
        self.db = db_manager
        self.config = config
        self.llm = create_llm(config)            # 大语言模型
        self.embeddings = create_embeddings(config)  # 向量嵌入模型
        self.top_k = config.get("app", {}).get("top_k", 3)

    def reload_config(self, config):
        """热更新配置"""
        self.config = config
        self.llm = create_llm(config)
        self.embeddings = create_embeddings(config)
        self.top_k = config.get("app", {}).get("top_k", 3)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """第一步：从知识库检索最相关的文档"""
        return self.db.search_similar(query, top_k or self.top_k)

    def format_docs(self, docs: List[Dict]) -> str:
        """把检索结果拼成一段上下文文本"""
        parts = []
        for i, d in enumerate(docs):
            source = d.get("metadata", {}).get("filename", "未知")
            parts.append(f"[来源: {source}]\n{d.get('content', '')}")
        return "\n\n".join(parts)

    def format_sources(self, docs: List[Dict]) -> List[Dict]:
        """提取来源信息（去重）"""
        sources = []
        seen = set()
        for d in docs:
            name = d.get("metadata", {}).get("filename", "")
            if name and name not in seen:
                seen.add(name)
                sources.append({"filename": name})
        return sources

    def generate(self, question: str, context: str, history: List[Dict] = None) -> str:
        """第二步：根据上下文让大模型生成回答"""
        prompt = self._build_prompt(history)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def generate_stream(self, question: str, context: str, history: List[Dict] = None):
        """流式生成（打字机效果）"""
        prompt = self._build_prompt(history)
        chain = prompt | self.llm | StrOutputParser()
        return chain.stream({"context": context, "question": question})

    def _build_prompt(self, history=None):
        """构建提示词，自动拼接最近5轮对话历史"""
        prompt_text = SYSTEM_PROMPT + "\n\n上下文信息：\n{context}"

        # 如果有对话历史，加进去让大模型能理解上下文
        if history:
            lines = []
            for h in history[-5:]:
                role = "用户" if h["role"] == "user" else "助手"
                lines.append(f"{role}: {h['content']}")
            history_text = "\n".join(lines)
            prompt_text = SYSTEM_PROMPT + f"\n\n对话历史：\n{history_text}\n\n上下文信息：\n{{context}}"

        return ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("human", "{question}"),
        ])
