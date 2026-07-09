"""
SmartKB 告警通知模块

本模块实现企业级告警通知功能，用于：
1. Agent工具调用死循环检测与熔断
2. API资费异常消耗告警
3. 系统异常实时通知

支持的告警渠道：
- 飞书: Webhook 富文本卡片
- 钉钉: Webhook Markdown
- 企业微信: Webhook Markdown

熔断机制：
- 当Agent工具调用次数超过阈值时触发熔断
- 自动终止当前任务，返回安全提示
- 异步发送告警通知到配置的渠道
"""

import os
import asyncio
from datetime import datetime
from typing import Optional

import httpx


# ============================================================================
# 告警配置（使用不可变配置 + 线程安全更新）
# ============================================================================

class AlertConfig:
    """告警配置（线程安全）"""

    def __init__(self):
        self._feishu_webhook = os.getenv("FEISHU_WEBHOOK_URL", "")
        self._dingtalk_webhook = os.getenv("DINGTALK_WEBHOOK_URL", "")
        self._wechat_webhook = os.getenv("WECHAT_WEBHOOK_URL", "")
        self._circuit_breaker_threshold: int = 5
        self._max_token_estimate: int = 20000

    @property
    def feishu_webhook(self) -> str:
        return self._feishu_webhook

    @property
    def dingtalk_webhook(self) -> str:
        return self._dingtalk_webhook

    @property
    def wechat_webhook(self) -> str:
        return self._wechat_webhook

    @property
    def circuit_breaker_threshold(self) -> int:
        return self._circuit_breaker_threshold

    @property
    def max_token_estimate(self) -> int:
        return self._max_token_estimate

    def update_circuit_breaker(self, threshold: int):
        self._circuit_breaker_threshold = threshold


# 全局配置实例
alert_config = AlertConfig()


# ============================================================================
# 飞书告警
# ============================================================================

async def send_feishu_alert(title: str, content: str, webhook_url: Optional[str] = None) -> bool:
    """发送飞书Webhook告警"""
    url = webhook_url or alert_config.feishu_webhook

    if not url:
        print("[告警] 飞书Webhook未配置，跳过")
        return False

    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"\U0001f6a8 {title}"},
                "template": "red",
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": content}},
                {
                    "tag": "note",
                    "elements": [
                        {"tag": "plain_text",
                         "content": f"SmartKB告警 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
                    ],
                },
            ],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                print(f"[告警] 飞书发送成功: {title}")
                return True
            print(f"[告警] 飞书发送失败: {response.text}")
            return False
    except Exception as e:
        print(f"[告警] 飞书异常: {str(e)}")
        return False


# ============================================================================
# 钉钉告警
# ============================================================================

async def send_dingtalk_alert(title: str, content: str, webhook_url: Optional[str] = None) -> bool:
    """发送钉钉Webhook告警"""
    url = webhook_url or alert_config.dingtalk_webhook

    if not url:
        print("[告警] 钉钉Webhook未配置，跳过")
        return False

    payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": title,
            "text": f"### \U0001f6a8 {title}\n\n{content}\n\n---\n*SmartKB告警系统*",
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload)
            return response.status_code == 200
    except Exception as e:
        print(f"[告警] 钉钉异常: {str(e)}")
        return False


# ============================================================================
# 企业微信告警
# ============================================================================

async def send_wechat_alert(title: str, content: str, webhook_url: Optional[str] = None) -> bool:
    """发送企业微信Webhook告警"""
    url = webhook_url or alert_config.wechat_webhook

    if not url:
        print("[告警] 企业微信Webhook未配置，跳过")
        return False

    payload = {
        "msgtype": "markdown",
        "markdown": {"content": f"### \U0001f6a8 {title}\n\n{content}"},
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload)
            return response.status_code == 200
    except Exception as e:
        print(f"[告警] 企业微信异常: {str(e)}")
        return False


# ============================================================================
# 统一告警入口
# ============================================================================

async def send_alert(title: str, content: str, platforms: list = None) -> list:
    """
    发送告警到多个平台

    并行发送到所有配置的平台，不阻塞主流程
    """
    if platforms is None:
        platforms = ["feishu", "dingtalk", "wechat"]

    tasks = []
    if "feishu" in platforms:
        tasks.append(send_feishu_alert(title, content))
    if "dingtalk" in platforms:
        tasks.append(send_dingtalk_alert(title, content))
    if "wechat" in platforms:
        tasks.append(send_wechat_alert(title, content))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return list(results)


# ============================================================================
# 熔断告警
# ============================================================================

def alert_circuit_breaker(session_id: str, tool_call_count: int, estimated_tokens: int = None):
    """
    触发熔断告警

    当Agent工具调用次数超过阈值时调用此函数。
    """
    if estimated_tokens is None:
        estimated_tokens = tool_call_count * 4000

    estimated_cost = estimated_tokens * 0.00002

    title = "Agent工具调用死循环熔断"
    content = (
        f"**会话ID**: `{session_id}`\n\n"
        f"**告警详情**:\n"
        f"- 工具调用次数: **{tool_call_count}次** (阈值: {alert_config.circuit_breaker_threshold}次)\n"
        f"- 预估Token消耗: **约{estimated_tokens:,}Token**\n"
        f"- 阻止损失: 约${estimated_cost:.2f}\n\n"
        f"**已执行操作**:\n"
        f"- ✅ 强制终止当前任务\n"
        f"- ✅ 返回安全提示信息\n\n"
        f"**建议**:\n"
        f"- 检查用户输入是否包含特殊字符\n"
        f"- 优化SQL生成提示词\n"
        f"- 考虑增加输入预处理过滤"
    )

    asyncio.create_task(send_alert(title, content))
    print(f"[熔断] 会话 {session_id} 触发熔断，调用 {tool_call_count} 次")


# ============================================================================
# API Key错误告警
# ============================================================================

def alert_api_key_error(provider: str, error_msg: str):
    """API Key配置错误告警"""
    title = "API Key配置异常"
    content = f"**云服务商**: {provider}\n\n**错误信息**: {error_msg}\n\n" \
              f"**建议操作**:\n- 检查.env文件中的API Key是否正确\n" \
              f"- 确认API Key是否已过期或余额不足\n- 验证网络连接是否正常"
    asyncio.create_task(send_alert(title, content))


# ============================================================================
# Token消耗告警
# ============================================================================

def alert_high_token_usage(session_id: str, token_count: int, threshold: int = 10000):
    """Token消耗过高告警"""
    if token_count >= threshold:
        title = "Token消耗过高"
        content = (
            f"**会话ID**: `{session_id}`\n\n"
            f"**消耗情况**:\n"
            f"- 本次会话Token消耗: **{token_count:,}Token**\n"
            f"- 告警阈值: {threshold:,}Token\n\n"
            f"**建议**:\n- 检查是否存在重复对话\n"
            f"- 考虑缩短对话历史长度"
        )
        asyncio.create_task(send_alert(title, content))


# ============================================================================
# 配置更新
# ============================================================================

def update_circuit_breaker_config(threshold: int = None):
    """更新熔断配置"""
    if threshold is not None:
        alert_config.update_circuit_breaker(threshold)
        print(f"[配置] 熔断阈值更新为: {threshold}")
