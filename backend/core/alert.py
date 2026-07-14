"""
告警通知 - 支持飞书/钉钉/企业微信
基于asyncio异步发送，不阻塞主业务流程
"""

import json
import asyncio
from typing import Optional
from urllib import request as urllib_request


class AlertManager:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.feishu_url = self.config.get("feishu_webhook", "")
        self.dingtalk_url = self.config.get("dingtalk_webhook", "")
        self.wechat_url = self.config.get("wechat_webhook", "")

    def send(self, title: str, content: str, level: str = "info"):
        """同步发送告警"""
        if self.feishu_url:
            self._send_feishu(title, content, level)
        if self.dingtalk_url:
            self._send_dingtalk(title, content, level)
        if self.wechat_url:
            self._send_wechat(title, content, level)

    def send_async(self, title: str, content: str, level: str = "info"):
        """异步发送告警（基于asyncio.create_task，不阻塞主流程）"""
        asyncio.create_task(self._send_all_async(title, content, level))

    async def _send_all_async(self, title: str, content: str, level: str = "info"):
        """异步发送到所有渠道"""
        tasks = []
        if self.feishu_url:
            tasks.append(self._async_post(self.feishu_url, self._build_feishu_payload(title, content, level)))
        if self.dingtalk_url:
            tasks.append(self._async_post(self.dingtalk_url, self._build_dingtalk_payload(title, content, level)))
        if self.wechat_url:
            tasks.append(self._async_post(self.wechat_url, self._build_wechat_payload(title, content, level)))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _async_post(self, url: str, data: dict):
        """异步HTTP POST"""
        try:
            payload = json.dumps(data, ensure_ascii=False).encode('utf-8')
            req = urllib_request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}
            )
            # 用asyncio线程池执行同步HTTP请求
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: urllib_request.urlopen(req, timeout=5))
        except Exception as e:
            print(f"[异步告警发送失败] {e}")

    def _post_json(self, url: str, data: dict):
        """同步发送JSON请求"""
        try:
            req = urllib_request.Request(
                url,
                data=json.dumps(data, ensure_ascii=False).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            urllib_request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[告警发送失败] {e}")

    def _build_feishu_payload(self, title, content, level):
        """构建飞书消息体"""
        color_map = {"info": "green", "warning": "yellow", "error": "red"}
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": color_map.get(level, "blue")
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "plain_text", "content": content}},
                    {"tag": "hr"},
                    {"tag": "note", "elements": [
                        {"tag": "plain_text", "content": f"级别: {level.upper()}"}
                    ]}
                ]
            }
        }

    def _build_dingtalk_payload(self, title, content, level):
        """构建钉钉消息体"""
        return {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": f"### {title}\n\n{content}"}
        }

    def _build_wechat_payload(self, title, content, level):
        """构建企业微信消息体"""
        return {
            "msgtype": "markdown",
            "markdown": {"content": f"### {title}\n{content}"}
        }

    def _send_feishu(self, title, content, level):
        """飞书机器人（同步）"""
        self._post_json(self.feishu_url, self._build_feishu_payload(title, content, level))

    def _send_dingtalk(self, title, content, level):
        """钉钉机器人（同步）"""
        self._post_json(self.dingtalk_url, self._build_dingtalk_payload(title, content, level))

    def _send_wechat(self, title, content, level):
        """企业微信机器人（同步）"""
        self._post_json(self.wechat_url, self._build_wechat_payload(title, content, level))
