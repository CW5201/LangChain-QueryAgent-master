# 配置管理
# 负责读取config.yaml，支持热更新（改完配置不用重启服务）

import os
import yaml

# 缓存配置，避免每次都读文件
_cache = None
_cache_time = 0


def load_config(path="backend/config.yaml"):
    """读取配置文件"""
    global _cache, _cache_time

    # 如果文件不存在，返回默认配置
    if not os.path.exists(path):
        return {
            "app": {"mode": "local", "chunk_size": 500, "chunk_overlap": 100, "top_k": 3},
            "local_model": {"model": "qwen2.5:7b", "base_url": "http://localhost:11434"},
            "cloud_model": {"model": "gpt-3.5-turbo", "api_key": ""},
            "agent": {"max_tool_calls": 5},
            "alert": {},
        }

    # 检查文件有没有变化，没变化就直接返回缓存
    mtime = os.path.getmtime(path)
    if _cache and mtime == _cache_time:
        return _cache

    # 读取yaml文件
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 转成内部使用的格式
    model = raw.get("model", {})
    rag = raw.get("rag", {})

    # 如果没有配置云端key，强制用local模式
    has_key = bool(model.get("cloud", {}).get("api_key"))

    _cache = {
        "app": {
            "mode": model.get("mode", "local") if has_key else "local",
            "chunk_size": rag.get("chunk_size", 500),
            "chunk_overlap": rag.get("chunk_overlap", 100),
            "top_k": rag.get("top_k", 3),
        },
        "local_model": {
            "model": model.get("local", {}).get("model_name", "qwen2.5:7b"),
            "base_url": model.get("local", {}).get("base_url", "http://localhost:11434"),
        },
        "cloud_model": {
            "model": model.get("cloud", {}).get("model_name", "gpt-3.5-turbo"),
            "api_key": model.get("cloud", {}).get("api_key", ""),
            "base_url": model.get("cloud", {}).get("base_url", ""),
        },
        "agent": {"max_tool_calls": raw.get("agent", {}).get("max_tool_calls", 5)},
        "alert": raw.get("alert", {}),
    }
    _cache_time = mtime
    return _cache


def reload_config(path="backend/config.yaml"):
    """强制重新读取配置（热更新用）"""
    global _cache, _cache_time
    _cache = None
    _cache_time = 0
    return load_config(path)
