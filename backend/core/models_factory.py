# 模型工厂
# 根据配置创建大语言模型和向量嵌入模型
# 支持两种模式：local（本地Ollama）和 cloud（云端API）

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def create_llm(config):
    """创建大语言模型"""
    mode = config.get("app", {}).get("mode", "local")

    if mode == "cloud":
        # 云端模式：用通义千问/DeepSeek/GPT等API
        cfg = config.get("cloud_model", {})
        return ChatOpenAI(
            model=cfg.get("model", "gpt-3.5-turbo"),
            api_key=cfg.get("api_key", ""),
            base_url=cfg.get("base_url", ""),
            temperature=0.3,
        )
    else:
        # 本地模式：用Ollama运行本地模型
        cfg = config.get("local_model", {})
        return ChatOllama(
            model=cfg.get("model", "qwen2.5:7b"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperature=0.3,
        )


def create_embeddings(config):
    """创建向量嵌入模型（把文本转成数字向量）"""
    mode = config.get("app", {}).get("mode", "local")

    if mode == "cloud":
        cfg = config.get("cloud_model", {})
        return OpenAIEmbeddings(
            model=cfg.get("embedding_model", "text-embedding-3-small"),
            api_key=cfg.get("api_key", ""),
            base_url=cfg.get("base_url", ""),
        )
    else:
        cfg = config.get("local_model", {})
        return OllamaEmbeddings(
            model=cfg.get("model", "qwen2.5:7b"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
        )
