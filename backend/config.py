"""
配置管理
"""

import os
import yaml


def load_config(path="backend/config.yaml"):
    """加载配置文件"""
    if not os.path.exists(path):
        # 返回默认配置
        return {
            "app": {"mode": "local", "chunk_size": 500, "chunk_overlap": 100},
            "local_model": {"model": "qwen2.5:7b", "base_url": "http://localhost:11434"},
            "cloud_model": {"model": "gpt-3.5-turbo", "api_key": "", "base_url": ""},
            "alert": {}
        }

    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    # 转换配置格式
    model_cfg = raw.get("model", {})
    rag_cfg = raw.get("rag", {})

    return {
        "app": {
            "mode": model_cfg.get("mode", "local") if model_cfg.get("cloud", {}).get("api_key") else "local",
            "chunk_size": rag_cfg.get("chunk_size", 500),
            "chunk_overlap": rag_cfg.get("chunk_overlap", 100),
        },
        "local_model": {
            "model": model_cfg.get("local", {}).get("model_name", "qwen2.5:7b"),
            "base_url": model_cfg.get("local", {}).get("base_url", "http://localhost:11434"),
            "embedding_model": model_cfg.get("local", {}).get("model_name", "qwen2.5:7b"),
        },
        "cloud_model": {
            "model": model_cfg.get("cloud", {}).get("model_name", "gpt-3.5-turbo"),
            "api_key": model_cfg.get("cloud", {}).get("api_key", ""),
            "base_url": "",
            "embedding_model": rag_cfg.get("embedding", {}).get("model", "text-embedding-v1"),
        },
        "alert": raw.get("alert", {}),
        "raw": raw,
    }
