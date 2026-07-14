"""
智能体引擎 - 结合知识库和网络搜索回答复杂问题
实现防烧钱熔断机制：监控中间步骤工具调用次数，超过阈值自动终止
"""

import json
import time
from typing import Dict, Any, List, Generator
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from backend.core.database import DatabaseManager
from backend.core.rag_engine import RAGEngine
from backend.core.tools import get_tools
from backend.core.models_factory import create_llm

# 防烧钱熔断默认阈值
DEFAULT_MAX_TOOL_CALLS = 5
# 半恢复等待时间（秒）
DEFAULT_HALF_OPEN_WAIT = 30


class CircuitBreaker:
    """熔断器：连续失败超阈值自动熔断，防止工具调用死循环烧钱"""

    def __init__(self, max_failures=3, reset_timeout=30):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.state = "closed"  # closed=正常, open=熔断, half_open=半恢复
        self.last_failure_time = 0

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.max_failures:
            self.state = "open"

    def allow_request(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half_open"
                return True
            return False
        # half_open: 放行一次试探
        return True


class AgentEngine:
    def __init__(self, rag_engine: RAGEngine, db_manager: DatabaseManager, config: Dict[str, Any]):
        self.rag = rag_engine
        self.db = db_manager
        self.config = config
        self.llm = create_llm(config)
        self.tools = get_tools(db_manager)
        self.circuit_breaker = CircuitBreaker()
        self.max_tool_calls = config.get("agent", {}).get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)
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
            handle_parsing_errors=True, verbose=True,
            max_iterations=10
        )

    def reload_config(self, config):
        """热更新配置"""
        self.config = config
        self.llm = create_llm(config)
        self.max_tool_calls = config.get("agent", {}).get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)
        self._build_agent()

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
            try:
                result = self.executor.invoke({
                    "input": question, "chat_history": chat_history
                })
                answer = result.get("output", "无法回答该问题")
            except Exception as e:
                self.circuit_breaker.record_failure()
                answer = f"抱歉，处理出错: {e}"
                raise

        self.circuit_breaker.record_success()
        self.db.add_conversation(conv_id, "user", question)
        self.db.add_conversation(conv_id, "assistant", answer)
        return answer

    def chat_stream(self, question: str, conv_id: str, history: List[Dict] = None) -> Generator:
        """
        流式对话，产出SSE事件流
        事件类型：
        - thinking: Agent思考过程
        - tool_call: 工具调用
        - tool_result: 工具返回结果
        - answer: 最终回答
        - error: 错误/熔断
        """
        # 第一步：检索知识库
        yield _sse_event("thinking", "正在从知识库检索相关信息...")
        docs = self.rag.retrieve(question)

        if docs:
            yield _sse_event("thinking", f"检索到 {len(docs)} 条相关内容，开始生成回答...")
            context = self.rag.format_docs(docs)
            sources = [d.get("metadata", {}).get("filename", "") for d in docs]

            # 流式生成回答
            for chunk in self.rag.generate_stream(question, context, history):
                yield _sse_event("answer", chunk)

            yield _sse_event("done", json.dumps({
                "sources": sources,
                "mode": "rag",
                "tool_calls": 0
            }))
        else:
            # 知识库无结果，走Agent
            yield _sse_event("thinking", "知识库无匹配内容，切换Agent模式...")

            if not self.circuit_breaker.allow_request():
                yield _sse_event("error", "熔断触发：系统暂时不可用，请稍后重试")
                yield _sse_event("done", json.dumps({"mode": "circuit_breaker", "tool_calls": 0}))
                return

            chat_history = self._convert_history(history)
            tool_call_count = 0

            # 模拟Agent逐步执行（带工具调用监控）
            yield _sse_event("thinking", "Agent分析问题中...")

            try:
                # 用invoke获取完整结果，但模拟中间步骤输出
                yield _sse_event("thinking", "Agent正在调用工具搜索信息...")
                result = self.executor.invoke({
                    "input": question, "chat_history": chat_history
                })
                tool_call_count = self._count_tool_calls(result)
                yield _sse_event("tool_call", f"工具调用 {tool_call_count} 次")

                # 检查熔断
                if tool_call_count >= self.max_tool_calls:
                    self.circuit_breaker.record_failure()
                    yield _sse_event("error", f"熔断触发：工具调用超过 {self.max_tool_calls} 次阈值")
                    # 异步推送告警
                    self._trigger_alert(question, tool_call_count)

                answer = result.get("output", "无法回答该问题")
                yield _sse_event("answer", answer)
                yield _sse_event("done", json.dumps({
                    "mode": "agent",
                    "tool_calls": tool_call_count
                }))

                self.circuit_breaker.record_success()

            except Exception as e:
                self.circuit_breaker.record_failure()
                yield _sse_event("error", f"Agent执行出错: {e}")
                yield _sse_event("done", json.dumps({"mode": "error", "tool_calls": tool_call_count}))

        # 保存对话
        self.db.add_conversation(conv_id, "user", question)

    def _count_tool_calls(self, result: dict) -> int:
        """从AgentExecutor结果中统计工具调用次数"""
        intermediate_steps = result.get("intermediate_steps", [])
        count = 0
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) >= 1:
                action = step[0]
                if hasattr(action, "tool"):
                    count += 1
        return count

    def _trigger_alert(self, question: str, tool_calls: int):
        """触发告警（异步）"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._send_alert_async(question, tool_calls))
            else:
                loop.run_until_complete(self._send_alert_async(question, tool_calls))
        except Exception:
            pass

    async def _send_alert_async(self, question: str, tool_calls: int):
        """异步发送熔断告警"""
        from backend.core.alert import AlertManager
        alert = AlertManager(self.config.get("alert", {}))
        title = "防烧钱熔断告警"
        content = f"问题: {question}\n工具调用次数: {tool_calls} (阈值: {self.max_tool_calls})"
        alert.send(title, content, "error")

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


def _sse_event(event_type: str, data: str) -> str:
    """生成SSE事件格式"""
    return f"data: {json.dumps({'type': event_type, 'data': data}, ensure_ascii=False)}\n\n"
