# Agent智能体引擎
# 功能：知识库没有答案时，自动调用工具搜索
# 防烧钱熔断：监控工具调用次数，超过阈值自动终止

import json
import time
from typing import Dict, List, Generator
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from backend.core.rag_engine import RAGEngine
from backend.core.tools import get_tools
from backend.core.models_factory import create_llm


class CircuitBreaker:
    """熔断器：防止工具调用死循环烧钱"""

    def __init__(self, max_failures=3, reset_timeout=30):
        self.max_failures = max_failures  # 连续失败几次就熔断
        self.reset_timeout = reset_timeout  # 熔断后等多久恢复
        self.failure_count = 0
        self.state = "closed"  # closed=正常, open=熔断, half_open=半恢复
        self.last_failure_time = 0

    def record_success(self):
        """记录成功：重置计数"""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """记录失败：计数+1，超限就熔断"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.max_failures:
            self.state = "open"

    def allow_request(self) -> bool:
        """判断是否允许请求"""
        if self.state == "closed":
            return True
        if self.state == "open":
            # 超过等待时间，尝试半恢复
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half_open"
                return True
            return False
        return True  # half_open状态放行一次


class AgentEngine:
    def __init__(self, rag_engine: RAGEngine, config: Dict):
        self.rag = rag_engine
        self.config = config
        self.llm = create_llm(config)
        self.tools = get_tools()
        self.max_tool_calls = config.get("agent", {}).get("max_tool_calls", 5)
        self.circuit_breaker = CircuitBreaker()
        self._build_agent()

    def _build_agent(self):
        """构建Agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是智能问答助手，可以使用database_query（知识库查询）和web_search（网络搜索）工具。优先用知识库。"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, handle_parsing_errors=True)

    def reload_config(self, config):
        """热更新配置"""
        self.config = config
        self.llm = create_llm(config)
        self.max_tool_calls = config.get("agent", {}).get("max_tool_calls", 5)
        self._build_agent()

    def chat(self, question: str, conv_id: str, history: List[Dict] = None) -> str:
        """普通对话：先检索知识库，没有就用Agent"""
        docs = self.rag.retrieve(question)
        if docs:
            context = self.rag.format_docs(docs)
            return self.rag.generate(question, context, history)

        # 知识库没结果，走Agent
        chat_history = self._to_messages(history)
        result = self.executor.invoke({"input": question, "chat_history": chat_history})
        return result.get("output", "无法回答该问题")

    def chat_stream(self, question: str, conv_id: str, history: List[Dict] = None) -> Generator:
        """流式对话：产出SSE事件流（thinking/tool_call/answer/done/error）"""
        # 先检索知识库
        yield _event("thinking", "正在检索知识库...")
        docs = self.rag.retrieve(question)

        if docs:
            # 知识库有结果，直接用RAG生成
            yield _event("thinking", f"找到 {len(docs)} 条相关内容")
            context = self.rag.format_docs(docs)
            for chunk in self.rag.generate_stream(question, context, history):
                yield _event("answer", chunk)
            yield _event("done", json.dumps({"mode": "rag", "tool_calls": 0, "sources": self.rag.format_sources(docs)}))
        else:
            # 知识库没结果，走Agent
            yield _event("thinking", "知识库无匹配，切换Agent模式...")

            # 检查熔断
            if not self.circuit_breaker.allow_request():
                yield _event("error", "熔断触发：系统暂时不可用，请稍后重试")
                yield _event("done", json.dumps({"mode": "circuit_breaker"}))
                return

            yield _event("thinking", "Agent正在调用工具...")
            try:
                result = self.executor.invoke({"input": question, "chat_history": self._to_messages(history)})
                tool_calls = len(result.get("intermediate_steps", []))
                yield _event("tool_call", f"工具调用 {tool_calls} 次")

                # 检查是否超阈值
                if tool_calls >= self.max_tool_calls:
                    self.circuit_breaker.record_failure()
                    yield _event("error", f"熔断：工具调用超过 {self.max_tool_calls} 次")
                    self._alert(question, tool_calls)
                else:
                    self.circuit_breaker.record_success()

                yield _event("answer", result.get("output", "无法回答"))
                yield _event("done", json.dumps({"mode": "agent", "tool_calls": tool_calls}))
            except Exception as e:
                self.circuit_breaker.record_failure()
                yield _event("error", f"Agent出错: {e}")
                yield _event("done", json.dumps({"mode": "error"}))

    def _to_messages(self, history):
        """把对话历史转成LangChain消息格式"""
        if not history:
            return []
        return [HumanMessage(content=h["content"]) if h["role"] == "user"
                else AIMessage(content=h["content"]) for h in history]

    def _alert(self, question, tool_calls):
        """触发告警（异步）"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._send_alert(question, tool_calls))
        except Exception:
            pass

    async def _send_alert(self, question, tool_calls):
        """异步发送熔断告警"""
        from backend.core.alert import AlertManager
        alert = AlertManager(self.config.get("alert", {}))
        alert.send("防烧钱熔断告警", f"问题: {question}\n调用次数: {tool_calls}", "error")


def _event(event_type, data):
    """生成SSE事件格式"""
    return f"data: {json.dumps({'type': event_type, 'data': data}, ensure_ascii=False)}\n\n"
