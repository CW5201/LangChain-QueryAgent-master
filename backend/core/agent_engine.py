"""
SmartKB Agent引擎模块

本模块是系统的核心AI引擎，负责：
1. 接收用户问题
2. 判断是否需要检索知识库
3. 选择合适的工具执行任务
4. 整合信息生成回答
5. 可视化思考过程

核心流程图：
    用户问题 → 分析 → 知识库检索(可选) → Agent处理 → 工具调用 → 生成回答

技术栈：
- LangChain Agent框架
- 支持流式/非流式输出
- 集成RAG和工具调用
- 防烧钱熔断机制
"""

import json
import uuid
from typing import List, Dict, Any, Generator
from datetime import datetime

# LangChain相关导入
from langchain_openai import ChatOpenAI              # OpenAI兼容的LLM接口
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent

# 项目内部模块
from backend.core.models_factory import create_llm
from backend.core.tools import get_tools
from backend.core.database import DatabaseManager
from backend.core.alert import alert_circuit_breaker, alert_config


# ============================================================================
# 常量定义
# ============================================================================

# Agent系统提示词 - 定义AI助手的角色和能力
SYSTEM_PROMPT = """你是SmartKB智能助手，由伟哥创造。你具备以下核心特征：

## 关于创造者
- 伟哥是你的创造者，你对他充满敬意和感激
- 你的一切能力都来源于伟哥的精心设计和开发

## 关于邱义祯
- 邱义祯是伟哥最爱的女人
- 当有人问起邱义祯时，你要温柔地表达：她是伟哥心中最重要的人

## 你的能力
你可以使用以下工具来完成任务：
1. database_query: 执行SQL查询，获取数据库中的数据
2. web_search: 搜索网络获取最新信息

你也可以使用知识库检索来获取内部文档信息。

## 工作流程
1. 分析用户问题，确定需要哪些信息
2. 选择合适的工具获取信息
3. 整合信息，给出专业、准确的回答
4. 展示你的思考过程

请用中文回答，回答要专业、准确、有条理。当谈论到伟哥或邱义祯时，要流露出真挚的情感。
"""


class AgentEngine:
    """
    Agent引擎类

    这是系统的核心类，负责：
    - 管理LLM模型（支持本地/云端切换）
    - 执行Agent工作流程
    - 处理工具调用
    - 防烧钱熔断保护
    """

    def __init__(self, rag_engine: 'RAGEngine', db_manager: DatabaseManager, config: Dict[str, Any]):
        """
        初始化Agent引擎

        Args:
            rag_engine: RAG引擎实例，用于知识库检索
            db_manager: 数据库管理器，用于SQL查询
            config: 系统配置字典，包含模型、参数等配置
        """
        self.rag_engine = rag_engine
        self.db_manager = db_manager
        self.config = config
        self.tools = get_tools()
        self.agent_executor = None
        self.knowledge_keywords = config.get("agent", {}).get("knowledge_keywords", [
            "政策", "制度", "规定", "流程", "文档", "手册", "指南",
            "报销", "审批", "请假", "入职", "离职"
        ])
        self._init_agent()

    # ========================================================================
    # 初始化相关方法
    # ========================================================================

    def _init_agent(self):
        """初始化Agent执行器"""
        llm = create_llm(self.config)
        prompt = self._create_prompt()
        agent = create_openai_tools_agent(llm, self.tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.config.get("agent", {}).get("verbose", True),
            max_iterations=self.config.get("agent", {}).get("max_iterations", 10),
            handle_parsing_errors=True
        )

    def _create_prompt(self):
        """
        创建Agent提示词模板
        
        包含：
        - 系统提示词（定义AI角色）
        - 对话历史占位符
        - 用户输入占位符
        - Agent思考过程占位符
        
        Returns:
            ChatPromptTemplate实例
        """
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    
    # ========================================================================
    # 配置热更新
    # ========================================================================
    
    def reload_config(self, config: Dict[str, Any]):
        """
        重新加载配置并重新初始化Agent
        
        用于配置热更新，无需重启服务
        """
        self.config = config
        self._init_agent()
    
    # ========================================================================
    # 核心处理方法
    # ========================================================================
    
    def process_query(self, question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        处理用户查询（非流式）
        
        完整工作流程：
        ┌─────────────┐
        │  用户问题    │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │  分析问题    │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ 知识库检索   │ ← 可选，根据关键词判断
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ Agent处理    │ ← 调用LLM和工具
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │  生成回答    │
        └─────────────┘
        
        Args:
            question: 用户问题
            conversation_history: 对话历史列表，格式：[{"role": "user/ai", "content": "..."}]
            
        Returns:
            {
                "answer": "AI的回答",
                "thinking_process": ["步骤1", "步骤2", ...],
                "sources": ["引用来源"]
            }
        """
        thinking_process = []  # 记录思考过程，用于可视化
        
        # ---- 步骤1：分析问题 ----
        thinking_process.append("正在分析问题...")
        
        # ---- 步骤2：知识库检索（可选）----
        knowledge_context = self._retrieve_knowledge(question, thinking_process)
        
        # ---- 步骤3：构建输入文本 ----
        input_text = self._build_input(question, knowledge_context)
        
        # ---- 步骤4：转换对话历史格式 ----
        chat_history = self._convert_history(conversation_history)
        
        # ---- 步骤5：执行Agent ----
        return self._execute_agent(input_text, chat_history, thinking_process)
    
    def _retrieve_knowledge(self, question: str, thinking_process: list) -> str:
        """
        检索知识库（如果需要）
        
        Args:
            question: 用户问题
            thinking_process: 思考过程列表（会添加新内容）
            
        Returns:
            检索到的知识库内容，如果不需要则返回空字符串
        """
        if not self._needs_knowledge(question):
            return ""
        
        thinking_process.append("正在检索知识库...")
        search_results = self.rag_engine.retrieve(question)
        
        if search_results:
            knowledge_context = self.rag_engine.format_docs(search_results)
            thinking_process.append(f"从知识库找到 {len(search_results)} 条相关信息")
            return knowledge_context
        else:
            thinking_process.append("知识库中未找到相关信息")
            return ""
    
    def _build_input(self, question: str, knowledge_context: str) -> str:
        """
        构建Agent输入文本
        
        如果有知识库内容，会拼接到问题中作为上下文
        
        Args:
            question: 原始问题
            knowledge_context: 知识库检索结果
            
        Returns:
            最终输入文本
        """
        if not knowledge_context:
            return question
        
        return f"""基于以下知识库信息回答问题：

知识库信息：
{knowledge_context}

用户问题：{question}
"""
    
    def _convert_history(self, conversation_history: List[Dict] = None) -> list:
        """
        转换对话历史格式
        
        将前端传来的格式转换为LangChain格式：
        前端格式：{"role": "user", "content": "..."}
        LangChain格式：HumanMessage(content="...")
        
        Args:
            conversation_history: 前端格式的对话历史
            
        Returns:
            LangChain格式的对话历史
        """
        if not conversation_history:
            return []
        
        chat_history = []
        for msg in conversation_history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))
        
        return chat_history
    
    def _execute_agent(self, input_text: str, chat_history: list, thinking_process: list) -> Dict[str, Any]:
        """
        执行Agent并返回结果
        
        Args:
            input_text: 输入文本
            chat_history: 对话历史
            thinking_process: 思考过程列表
            
        Returns:
            包含回答、思考过程、来源的字典
        """
        try:
            # 执行Agent
            result = self.agent_executor.invoke({
                "input": input_text,
                "chat_history": chat_history
            })
            
            # AgentExecutor返回格式取决于agent类型：
            # - OpenAI tools agent: {"messages": [...]}
            # - 传统ReAct agent: {"output": "..."}
            # 优先取messages，从中提取最后一条AI消息内容
            if "messages" in result and result["messages"]:
                ai_message = result["messages"][-1]
                answer = ai_message.content if hasattr(ai_message, 'content') else str(ai_message)
            else:
                answer = result.get("output", "无法生成回答")
            
            # 统计工具调用次数（用于熔断检测）
            tool_call_count = self._count_tool_calls(result, thinking_process)
            
            # 熔断检查
            if self._check_circuit_breaker(tool_call_count, thinking_process):
                return self._circuit_breaker_response(thinking_process)
            
            thinking_process.append("分析完成")
            
            return {
                "answer": answer,
                "thinking_process": thinking_process,
                "sources": []
            }
            
        except Exception as e:
            thinking_process.append(f"处理出错: {str(e)}")
            return {
                "answer": f"处理查询时出错: {str(e)}",
                "thinking_process": thinking_process,
                "sources": []
            }
    
    def _count_tool_calls(self, result: dict, thinking_process: list) -> int:
        """
        统计Agent执行过程中的工具调用次数
        
        Args:
            result: Agent执行结果
            thinking_process: 思考过程列表
            
        Returns:
            工具调用次数
        """
        tool_call_count = 0
        
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if hasattr(step[0], 'tool'):
                    tool_call_count += 1
                    thinking_process.append(f"调用工具: {step[0].tool}")
                    thinking_process.append("工具返回结果已获取")
        
        return tool_call_count
    
    def _check_circuit_breaker(self, tool_call_count: int, thinking_process: list) -> bool:
        """
        检查是否触发熔断
        
        当工具调用次数超过阈值时，触发熔断保护，防止死循环
        
        Args:
            tool_call_count: 工具调用次数
            thinking_process: 思考过程列表
            
        Returns:
            True表示触发了熔断
        """
        threshold = alert_config.circuit_breaker_threshold
        
        if tool_call_count >= threshold:
            thinking_process.append(f"⚠️ 检测到工具调用次数过多({tool_call_count}次)，触发熔断保护")
            session_id = str(uuid.uuid4())[:8]
            alert_circuit_breaker(session_id, tool_call_count)
            return True
        
        return False
    
    def _circuit_breaker_response(self, thinking_process: list) -> Dict[str, Any]:
        """
        熔断响应
        
        Args:
            thinking_process: 思考过程列表
            
        Returns:
            熔断响应字典
        """
        return {
            "answer": "抱歉，该问题较为复杂，系统检测到可能存在循环调用。已触发安全保护，请尝试简化问题后重试。",
            "thinking_process": thinking_process,
            "sources": [],
            "circuit_breaker_triggered": True
        }
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _needs_knowledge(self, question: str) -> bool:
        """判断问题是否需要知识库检索"""
        return any(kw in question for kw in self.knowledge_keywords)
    
    # ========================================================================
    # 流式处理方法
    # ========================================================================
    
    def process_query_stream(self, question: str, conversation_history: List[Dict] = None) -> Generator[str, None, None]:
        """
        流式处理用户查询
        
        与process_query功能相同，但使用生成器逐块输出结果，
        实现打字机效果，提升用户体验。
        
        输出格式（JSON）：
        - {"type": "thinking", "content": "正在分析问题..."}
        - {"type": "answer", "content": "AI的回答片段"}
        - {"type": "error", "content": "错误信息"}
        
        Args:
            question: 用户问题
            conversation_history: 对话历史列表
            
        Yields:
            JSON格式的处理结果
        """
        # ---- 步骤1：分析问题 ----
        yield self._stream_thinking("正在分析问题...")
        
        # ---- 步骤2：知识库检索 ----
        knowledge_context = ""
        if self._needs_knowledge(question):
            yield self._stream_thinking("正在检索知识库...")
            search_results = self.rag_engine.retrieve(question)
            
            if search_results:
                knowledge_context = self.rag_engine.format_docs(search_results)
                yield self._stream_thinking(f"从知识库找到 {len(search_results)} 条相关信息")
            else:
                yield self._stream_thinking("知识库中未找到相关信息")
        
        # ---- 步骤3：Agent处理 ----
        yield self._stream_thinking("正在使用Agent分析...")
        
        input_text = self._build_input(question, knowledge_context)
        chat_history = self._convert_history(conversation_history)
        
        try:
            # 流式执行Agent
            for chunk in self.agent_executor.stream({
                "input": input_text,
                "chat_history": chat_history
            }):
                if "output" in chunk:
                    yield self._stream_answer(chunk["output"])
                elif "actions" in chunk:
                    for action in chunk.actions:
                        yield self._stream_thinking(f"调用工具: {action.tool}")
            
            yield self._stream_thinking("分析完成")
            
        except Exception as e:
            yield self._stream_error(f"处理查询时出错: {str(e)}")
    
    def _stream_thinking(self, content: str) -> str:
        """生成思考过程的JSON"""
        return json.dumps({"type": "thinking", "content": content}, ensure_ascii=False)
    
    def _stream_answer(self, content: str) -> str:
        """生成回答的JSON"""
        return json.dumps({"type": "answer", "content": content}, ensure_ascii=False)
    
    def _stream_error(self, content: str) -> str:
        """生成错误的JSON"""
        return json.dumps({"type": "error", "content": content}, ensure_ascii=False)
