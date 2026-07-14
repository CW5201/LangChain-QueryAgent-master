"""
智能体引擎 - 结合知识库和网络搜索回答复杂问题
"""

from typing import Dict, Any, List
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from backend.core.database import DatabaseManager
from backend.core.rag_engine import RAGEngine
from backend.core.tools import get_tools
from backend.core.models_factory import create_llm


class AgentEngine:
    def __init__(self, rag_engine: RAGEngine, db_manager: DatabaseManager, config: Dict[str, Any]):
        self.rag = rag_engine
        self.db = db_manager
        self.config = config
        self.llm = create_llm(config)
        self.agent = None
        self.tools = get_tools(db_manager)
        self._build_agent()

    def _build_agent(self):
        """构建工具调用智能体"""
        system_msg = """你是一个智能问答助手，可以使用以下工具：
- database_query: 从知识库查询信息
- web_search: 从网络搜索最新信息
请根据问题判断使用哪个工具，优先使用知识库。"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=self.agent, tools=self.tools,
            handle_parsing_errors=True, verbose=True
        )

    def chat(self, question: str, conv_id: str, history: List[Dict] = None) -> str:
        """执行对话，先检索再回答"""
        # 先尝试从知识库检索
        docs = self.rag.retrieve(question)
        if docs:
            context = self.rag.format_docs(docs)
            answer = self.rag.generate(question, context, history)
        else:
            # 知识库没有相关内容，用Agent搜索
            chat_history = self._convert_history(history)
            result = self.executor.invoke({
                "input": question, "chat_history": chat_history
            })
            answer = result.get("output", "无法回答该问题")

        # 保存对话记录
        self.db.add_conversation(conv_id, "user", question)
        self.db.add_conversation(conv_id, "assistant", answer)
        return answer

    def _convert_history(self, history: List[Dict] = None):
        """把历史记录转成Agent需要的格式"""
        if not history:
            return []
        messages = []
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(AIMessage(content=h["content"]))
        return messages
