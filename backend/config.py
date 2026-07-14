"""
配置管理 - 支持热更新，运行时修改config.yaml无需重启
"""

import os
import yaml
import time


# 全局配置缓存
_config_cache = None
_config_mtime = 0


def load_config(path="backend/config.yaml"):
    """加载配置文件，自动检测文件修改时间实现热更新"""
    global _config_cache, _config_mtime

    if not os.path.exists(path):
        return _default_config()

    mtime = os.path.getmtime(path)
    if _config_cache is not None and mtime == _config_mtime:
        return _config_cache

    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    _config_cache = _parse_config(raw)
    _config_mtime = mtime
    return _config_cache


def reload_config(path="backend/config.yaml"):
    """强制重新加载配置（用于热更新接口）"""
    global _config_cache, _config_mtime
    _config_cache = None
    _config_mtime = 0
    return load_config(path)


def save_config(config: dict, path="backend/config.yaml"):
    """保存配置到文件"""
    raw = {
        "model": {
            "mode": config.get("app", {}).get("mode", "local"),
            "cloud": {
                "provider": "openai",
                "model_name": config.get("cloud_model", {}).get("model", "gpt-3.5-turbo"),
                "api_key": config.get("cloud_model", {}).get("api_key", ""),
                "base_url": config.get("cloud_model", {}).get("base_url", ""),
            },
            "local": {
                "model_name": config.get("local_model", {}).get("model", "qwen2.5:7b"),
                "base_url": config.get("local_model", {}).get("base_url", "http://localhost:11434"),
            }
        },
        "rag": {
            "chunk_size": config.get("app", {}).get("chunk_size", 500),
            "chunk_overlap": config.get("app", {}).get("chunk_overlap", 100),
            "embedding": {
                "model": config.get("cloud_model", {}).get("embedding_model", "text-embedding-v1"),
                "dimension": 1536
            },
            "top_k": config.get("app", {}).get("top_k", 3),
        },
        "agent": {
            "max_tool_calls": config.get("agent", {}).get("max_tool_calls", 5),
            "verbose": True,
        },
        "alert": config.get("alert", {}),
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
        }
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(raw, f, allow_unicode=True, default_flow_style=False)

    # 清除缓存
    global _config_cache, _config_mtime
    _config_cache = None
    _config_mtime = 0


def _parse_config(raw: dict) -> dict:
    """将yaml原始配置转成内部统一格式"""
    model_cfg = raw.get("model", {})
    rag_cfg = raw.get("rag", {})
    agent_cfg = raw.get("agent", {})

    has_cloud_key = bool(model_cfg.get("cloud", {}).get("api_key"))
    mode = model_cfg.get("mode", "local") if has_cloud_key else "local"

    return {
        "app": {
            "mode": mode,
            "chunk_size": rag_cfg.get("chunk_size", 500),
            "chunk_overlap": rag_cfg.get("chunk_overlap", 100),
            "top_k": rag_cfg.get("top_k", 3),
        },
        "local_model": {
            "model": model_cfg.get("local", {}).get("model_name", "qwen2.5:7b"),
            "base_url": model_cfg.get("local", {}).get("base_url", "http://localhost:11434"),
            "embedding_model": model_cfg.get("local", {}).get("model_name", "qwen2.5:7b"),
        },
        "cloud_model": {
            "model": model_cfg.get("cloud", {}).get("model_name", "gpt-3.5-turbo"),
            "api_key": model_cfg.get("cloud", {}).get("api_key", ""),
            "base_url": model_cfg.get("cloud", {}).get("base_url", ""),
            "embedding_model": rag_cfg.get("embedding", {}).get("model", "text-embedding-v1"),
        },
        "agent": {
            "max_tool_calls": agent_cfg.get("max_tool_calls", 5),
            "verbose": agent_cfg.get("verbose", True),
        },
        "alert": raw.get("alert", {}),
        "raw": raw,
    }


def _default_config() -> dict:
    """默认配置"""
    return {
        "app": {"mode": "local", "chunk_size": 500, "chunk_overlap": 100, "top_k": 3},
        "local_model": {"model": "qwen2.5:7b", "base_url": "http://localhost:11434", "embedding_model": "qwen2.5:7b"},
        "cloud_model": {"model": "gpt-3.5-turbo", "api_key": "", "base_url": "", "embedding_model": "text-embedding-v1"},
        "agent": {"max_tool_calls": 5, "verbose": True},
        "alert": {},
    }
