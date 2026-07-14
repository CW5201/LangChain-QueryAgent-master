"""
模型工厂 - 根据配置创建LLM和Embeddings
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def create_llm(config: dict):
    """创建语言模型"""
    app_cfg = config.get("app", {})
    mode = app_cfg.get("mode", "local")

    if mode == "cloud":
        model_cfg = config.get("cloud_model", {})
        return ChatOpenAI(
            model=model_cfg.get("model", "gpt-3.5-turbo"),
            api_key=model_cfg.get("api_key", ""),
            base_url=model_cfg.get("base_url", ""),
            temperature=0.3
        )
    else:
        model_cfg = config.get("local_model", {})
        return ChatOllama(
            model=model_cfg.get("model", "qwen2.5:7b"),
            base_url=model_cfg.get("base_url", "http://localhost:11434"),
            temperature=0.3
        )


def create_embeddings(config: dict):
    """创建向量嵌入模型"""
    app_cfg = config.get("app", {})
    mode = app_cfg.get("mode", "local")

    if mode == "cloud":
        model_cfg = config.get("cloud_model", {})
        return OpenAIEmbeddings(
            model=model_cfg.get("embedding_model", "text-embedding-3-small"),
            api_key=model_cfg.get("api_key", ""),
            base_url=model_cfg.get("base_url", "")
        )
    else:
        model_cfg = config.get("local_model", {})
        return OllamaEmbeddings(
            model=model_cfg.get("embedding_model", "qwen2.5:7b"),
            base_url=model_cfg.get("base_url", "http://localhost:11434")
        )
