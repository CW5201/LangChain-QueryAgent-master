"""
SmartKB 模型工厂模块

统一管理 LLM 和 Embedding 模型的创建，消除 agent_engine 和 rag_engine
之间的重复代码。

使用方式:
    from backend.core.models_factory import create_llm, create_embeddings

    llm = create_llm(config)
    emb = create_embeddings(config)
"""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings


# ============================================================================
# 常量定义
# ============================================================================

DEFAULT_CLOUD_LLM = "qwen-turbo"
DEFAULT_DEESEEK_LLM = "deepseek-chat"
DEFAULT_LOCAL_LLM = "qwen2.5:7b"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v1"
DEFAULT_LOCAL_EMBEDDING_MODEL = "nomic-embed-text"
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
OLLAMA_DEFAULT_URL = "http://localhost:11434"


def _get_api_key(env_var: str, fallback: str = "") -> str:
    """安全获取 API Key，优先环境变量"""
    return os.getenv(env_var, fallback)


# ============================================================================
# LLM 创建
# ============================================================================

def create_llm(config: dict) -> ChatOpenAI:
    """
    根据配置创建 LLM 实例

    Args:
        config: 系统配置字典，包含 model / llm 两个层级

    Returns:
        ChatOpenAI 实例
    """
    model_config = config.get("model", {})
    llm_config = config.get("llm", {})
    mode = model_config.get("mode", "local")
    temperature = llm_config.get("temperature", 0.7)
    streaming = llm_config.get("streaming", True)

    if mode == "cloud":
        return _create_cloud_llm(model_config, temperature, streaming)
    return _create_local_llm(model_config, temperature, streaming)


def _create_cloud_llm(model_config: dict, temperature: float, streaming: bool) -> ChatOpenAI:
    provider = model_config.get("cloud", {}).get("provider", "dashscope")
    api_key = _get_api_key("DASHSCOPE_API_KEY") or model_config.get("cloud", {}).get("api_key", "")

    if provider == "dashscope":
        return ChatOpenAI(
            model=model_config.get("cloud", {}).get("model_name", DEFAULT_CLOUD_LLM),
            api_key=api_key,
            base_url=DASHSCOPE_URL,
            temperature=temperature,
            streaming=streaming,
        )

    # DeepSeek
    api_key = _get_api_key("DEEPSEEK_API_KEY") or api_key
    return ChatOpenAI(
        model=model_config.get("cloud", {}).get("model_name", DEFAULT_DEESEEK_LLM),
        api_key=api_key,
        base_url=DEEPSEEK_URL,
        temperature=temperature,
        streaming=streaming,
    )


def _create_local_llm(model_config: dict, temperature: float, streaming: bool) -> ChatOpenAI:
    base_url = model_config.get("local", {}).get("base_url", OLLAMA_DEFAULT_URL)
    return ChatOpenAI(
        model=model_config.get("local", {}).get("model_name", DEFAULT_LOCAL_LLM),
        api_key="ollama",
        base_url=f"{base_url}/v1",
        temperature=temperature,
        streaming=streaming,
    )


# ============================================================================
# Embedding 创建
# ============================================================================

def create_embeddings(config: dict) -> object:
    """
    根据配置创建 Embedding 实例

    Args:
        config: 系统配置字典，包含 model / rag 两个层级

    Returns:
        Embedding 实例
    """
    model_config = config.get("model", {})
    rag_config = config.get("rag", {})
    mode = model_config.get("mode", "local")

    if mode == "cloud":
        return _create_cloud_embeddings(model_config, rag_config)
    return _create_local_embeddings(model_config)


def _create_cloud_embeddings(model_config: dict, rag_config: dict) -> OpenAIEmbeddings:
    api_key = _get_api_key("DASHSCOPE_API_KEY") or model_config.get("cloud", {}).get("api_key", "")
    return OpenAIEmbeddings(
        model=rag_config.get("embedding", {}).get("model", DEFAULT_EMBEDDING_MODEL),
        api_key=api_key,
        base_url=DASHSCOPE_URL,
    )


def _create_local_embeddings(model_config: dict) -> OllamaEmbeddings:
    base_url = model_config.get("local", {}).get("base_url", OLLAMA_DEFAULT_URL)
    return OllamaEmbeddings(
        model=DEFAULT_LOCAL_EMBEDDING_MODEL,
        base_url=base_url,
    )
