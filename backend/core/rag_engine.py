"""
SmartKB RAG（检索增强生成）引擎模块

RAG是什么？
    RAG = Retrieval Augmented Generation（检索增强生成）
    简单说就是：先从知识库找到相关资料，再让AI基于这些资料回答问题
    
工作流程图：
    ┌─────────────┐
    │  用户提问    │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 问题向量化   │ ← 将文本转换为数字向量
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 向量检索    │ ← 在ChromaDB中找最相似的文档
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 拼接上下文   │ ← 将检索到的文档拼接成上下文
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ LLM生成回答  │ ← AI基于上下文生成回答
    └─────────────┘

技术栈：
- 嵌入模型: 通义千问(text-embedding-v1) 或 Ollama(nomic-embed-text)
- 向量数据库: ChromaDB
- LLM: 通义千问/DeepSeek/Ollama
"""

from typing import List, Dict, Any

# LangChain相关导入
from langchain_chroma import Chroma                          # ChromaDB向量存储
from langchain_core.prompts import ChatPromptTemplate        # 提示词模板
from langchain_core.output_parsers import StrOutputParser    # 输出解析器

# 项目内部模块
from backend.core.database import DatabaseManager
from backend.core.models_factory import create_llm, create_embeddings


# ============================================================================
# RAG提示词模板
# ============================================================================

RAG_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据以下提供的上下文信息回答问题。

规则：
1. 仅基于提供的上下文信息回答问题
2. 如果上下文中没有相关信息，请明确说明"根据现有知识库内容，无法回答这个问题"
3. 回答要准确、简洁、专业
4. 如果可能，请引用来源文档名称

上下文信息：
{context}
"""


class RAGEngine:
    """
    RAG检索增强生成引擎
    
    核心功能：
    1. retrieve() - 从知识库检索相关文档
    2. format_docs() - 格式化检索结果
    3. generate() - 基于上下文生成回答
    
    使用示例：
        rag = RAGEngine(db_manager, config)
        results = rag.retrieve("如何报销差旅费？")
        context = rag.format_docs(results)
        answer = rag.generate("如何报销差旅费？", context)
    """
    
    def __init__(self, db_manager: DatabaseManager, config: Dict[str, Any]):
        """
        初始化RAG引擎
        
        Args:
            db_manager: 数据库管理器（管理SQLite和ChromaDB）
            config: 系统配置字典
        """
        self.db_manager = db_manager
        self.config = config
        self.llm = None          # 大语言模型（用于生成回答）
        self.embeddings = None   # 嵌入模型（用于将文本转为向量）
        self._init_components()
    
    # ========================================================================
    # 初始化方法
    # ========================================================================
    
    def _init_components(self):
        """
        初始化LLM和嵌入模型

        根据配置文件选择：
        - mode="cloud": 使用云端API（通义千问/DeepSeek）
        - mode="local": 使用本地Ollama服务
        """
        self.llm = create_llm(self.config)
        self.embeddings = create_embeddings(self.config)

    # ========================================================================
    # 配置热更新
    # ========================================================================
    
    def reload_config(self, config: Dict[str, Any]):
        """重新加载配置并重新初始化组件"""
        self.config = config
        self._init_components()
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        语义检索相关文档
        
        原理：
        1. 将用户问题转换为向量
        2. 在ChromaDB中计算与所有文档的相似度
        3. 返回最相似的top_k个文档
        
        Args:
            query: 用户查询文本
            top_k: 返回最相似的K个结果（默认3个）
            
        Returns:
            搜索结果列表，格式：
            [
                {"content": "文档内容", "metadata": {"filename": "xxx.pdf"}, "score": 0.85},
                ...
            ]
        """
        return self.db_manager.search_similar(query, top_k)
    
    def format_docs(self, docs: List[Dict]) -> str:
        """
        格式化检索到的文档
        
        将多个文档分块格式化为单个上下文字符串，
        包含来源信息，便于LLM理解和引用
        
        输入格式：
            [{"content": "...", "metadata": {"filename": "xxx.pdf"}}, ...]
            
        输出格式：
            [来源: xxx.pdf]
            文档内容1
            
            [来源: yyy.pdf]
            文档内容2
        
        Args:
            docs: 检索结果列表
            
        Returns:
            格式化后的上下文字符串
        """
        formatted = []
        
        for doc in docs:
            source = doc.get("metadata", {}).get("filename", "未知文档")
            content = doc.get("content", "")
            formatted.append(f"[来源: {source}]\n{content}")
        
        return "\n\n".join(formatted)
    
    def generate(self, question: str, context: str, conversation_history: List[Dict] = None) -> str:
        """
        基于上下文生成回答
        
        使用LLM根据检索到的上下文信息回答用户问题。
        
        Args:
            question: 用户问题
            context: 检索到的上下文信息（来自format_docs）
            conversation_history: 对话历史（可选）
            
        Returns:
            生成的回答文本
        """
        # 构建完整的系统提示词（包含对话历史）
        system_prompt = self._build_prompt_with_history(conversation_history)
        
        # 创建提示词模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        # 创建处理链: 提示词 → LLM → 字符串输出解析器
        # 这是LangChain的LCEL语法，用 | 连接各个组件
        chain = prompt | self.llm | StrOutputParser()
        
        # 调用链生成回答
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        return answer
    
    def generate_stream(self, question: str, context: str, conversation_history: List[Dict] = None):
        """
        流式生成回答
        
        与generate()功能相同，但使用生成器逐块输出，
        实现打字机效果。
        
        Args:
            question: 用户问题
            context: 检索到的上下文信息
            conversation_history: 对话历史（可选）
            
        Yields:
            生成的文本片段
        """
        system_prompt = self._build_prompt_with_history(conversation_history)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        # 流式调用，返回生成器
        return chain.stream({
            "context": context,
            "question": question
        })
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _build_prompt_with_history(self, conversation_history: List[Dict] = None) -> str:
        """
        构建包含对话历史的系统提示词
        
        如果有对话历史，会追加到提示词中，
        让AI能够理解上下文进行多轮对话。
        
        Args:
            conversation_history: 对话历史列表
            
        Returns:
            完整的系统提示词
        """
        prompt = RAG_SYSTEM_PROMPT
        
        if conversation_history:
            # 只保留最近5轮对话（避免token过多）
            history_text = "\n".join([
                f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
                for msg in conversation_history[-5:]
            ])
            prompt += f"\n\n对话历史：\n{history_text}"
        
        return prompt
