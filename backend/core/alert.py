"""
告警通知 - 支持飞书/钉钉/企业微信
"""

import json
from typing import Optional
from urllib import request as urllib_request


class AlertManager:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.feishu_url = self.config.get("feishu_webhook", "")
        self.dingtalk_url = self.config.get("dingtalk_webhook", "")
        self.wechat_url = self.config.get("wechat_webhook", "")

    def send(self, title: str, content: str, level: str = "info"):
        """发送告警到所有配置的渠道"""
        if self.feishu_url:
            self._send_feishu(title, content, level)
        if self.dingtalk_url:
            self._send_dingtalk(title, content, level)
        if self.wechat_url:
            self._send_wechat(title, content, level)

    def _post_json(self, url: str, data: dict):
        """发送JSON请求"""
        try:
            req = urllib_request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            urllib_request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[告警发送失败] {e}")

    def _send_feishu(self, title, content, level):
        """飞书机器人"""
        color_map = {"info": "green", "warning": "yellow", "error": "red"}
        data = {
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": title}},
                "elements": [
                    {"tag": "div", "text": {"tag": "plain_text", "content": content}},
                    {"tag": "hr"},
                    {"tag": "note", "elements": [
                        {"tag": "plain_text", "content": f"级别: {level.upper()}"}
                    ]}
                ]
            }
        }
        self._post_json(self.feishu_url, data)

    def _send_dingtalk(self, title, content, level):
        """钉钉机器人"""
        data = {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": f"### {title}\n\n{content}"}
        }
        self._post_json(self.dingtalk_url, data)

    def _send_wechat(self, title, content, level):
        """企业微信机器人"""
        data = {
            "msgtype": "markdown",
            "markdown": {"content": f"### {title}\n{content}"}
        }
        self._post_json(self.wechat_url, data)
