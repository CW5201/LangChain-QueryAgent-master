# 告警通知模块
# 支持飞书、钉钉、企业微信三种渠道
# 用asyncio异步发送，不阻塞主业务

import json
import asyncio
from urllib import request as urllib_request


class AlertManager:
    def __init__(self, config=None):
        config = config or {}
        self.feishu_url = config.get("feishu_webhook", "")
        self.dingtalk_url = config.get("dingtalk_webhook", "")
        self.wechat_url = config.get("wechat_webhook", "")

    def send(self, title, content, level="info"):
        """同步发送告警"""
        if self.feishu_url:
            self._post(self.feishu_url, self._feishu_payload(title, content, level))
        if self.dingtalk_url:
            self._post(self.dingtalk_url, self._dingtalk_payload(title, content))
        if self.wechat_url:
            self._post(self.wechat_url, self._wechat_payload(title, content))

    def send_async(self, title, content, level="info"):
        """异步发送（不阻塞主流程）"""
        asyncio.create_task(self._send_all(title, content, level))

    async def _send_all(self, title, content, level):
        """异步发到所有渠道"""
        tasks = []
        if self.feishu_url:
            tasks.append(self._async_post(self.feishu_url, self._feishu_payload(title, content, level)))
        if self.dingtalk_url:
            tasks.append(self._async_post(self.dingtalk_url, self._dingtalk_payload(title, content)))
        if self.wechat_url:
            tasks.append(self._async_post(self.wechat_url, self._wechat_payload(title, content)))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _post(self, url, data):
        """同步HTTP POST"""
        try:
            req = urllib_request.Request(
                url, data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            urllib_request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[告警失败] {e}")

    async def _async_post(self, url, data):
        """异步HTTP POST"""
        try:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            req = urllib_request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: urllib_request.urlopen(req, timeout=5))
        except Exception as e:
            print(f"[异步告警失败] {e}")

    def _feishu_payload(self, title, content, level):
        """飞书消息格式"""
        color = {"info": "green", "warning": "yellow", "error": "red"}.get(level, "blue")
        return {
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": title}, "template": color},
                "elements": [
                    {"tag": "div", "text": {"tag": "plain_text", "content": content}},
                    {"tag": "hr"},
                    {"tag": "note", "elements": [{"tag": "plain_text", "content": f"级别: {level.upper()}"}]}
                ]
            }
        }

    def _dingtalk_payload(self, title, content):
        """钉钉消息格式"""
        return {"msgtype": "markdown", "markdown": {"title": title, "text": f"### {title}\n\n{content}"}}

    def _wechat_payload(self, title, content):
        """企业微信消息格式"""
        return {"msgtype": "markdown", "markdown": {"content": f"### {title}\n{content}"}}
