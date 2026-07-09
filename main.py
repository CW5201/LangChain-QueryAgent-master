"""
AI 知识库问答系统 — 单文件版本
功能：上传文档 → RAG检索 → 大模型生成答案
运行: python app.py → 浏览器打开 http://localhost:7860
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ============ 加载配置 ============
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请在 .env 文件中配置 DASHSCOPE_API_KEY")

EMBEDDING_MODEL = "text-embedding-v3"
LLM_MODEL = "qwen-turbo"
VECTOR_STORE_DIR = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ============ 模型初始化 ============
def _get_embeddings():
    """获取 Embedding 模型"""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=EMBEDDING_MODEL,
    )


def _get_llm():
    """获取大语言模型"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=LLM_MODEL,
        temperature=0.3,
    )


# ============ 向量数据库 ============
def _get_vectorstore():
    """创建或加载持久化的向量数据库"""
    import chromadb
    emb = _get_embeddings()
    if os.path.exists(VECTOR_STORE_DIR):
        db = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        collection = db.get_or_create_collection("knowledge")
        return emb, db, collection
    else:
        db = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        collection = db.create_collection("knowledge")
        print("🆕 创建新的向量数据库")
        return emb, db, collection


# ============ 文档处理 ============
def _split_text(text):
    """文本切分"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )
    return splitter.split_text(text)


def _load_document(file_path):
    """根据文件类型加载文档"""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        texts = [page.extract_text() or "" for page in reader.pages]
        return [t for t in texts if t.strip()]
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return [f.read()]
    else:
        raise ValueError(f"不支持的文件格式: {ext}，请使用 .pdf 或 .txt")


def _add_to_vectorstore(file_path, emb, collection):
    """加载文件 → 切分 → 向量化 → 存入数据库"""
    texts = _load_document(file_path)
    if not texts:
        print(f"  ⚠️  文件为空: {file_path}")
        return 0

    chunks = []
    metadatas = []
    for i, text in enumerate(texts):
        sub_chunks = _split_text(text)
        for j, chunk in enumerate(sub_chunks):
            chunks.append(chunk)
            metadatas.append({"source": Path(file_path).name, "page": j})

    print(f"  📄 {len(texts)} 段文本，切分为 {len(chunks)} 块")

    if chunks:
        collection.add(documents=chunks, metadatas=metadatas, ids=[str(i) for i in range(len(chunks))])

    return len(chunks)


# ============ RAG 检索 + 生成 ============
def _ask_rag(question, emb, collection):
    """RAG 核心：检索相关文档 → 拼接上下文 → 大模型生成答案"""
    # 1. 检索最相关的文档块
    results = collection.query(query_texts=[question], n_results=4, include=["documents", "metadatas"])
    documents = results["documents"][0] if results["documents"] else []

    if not documents:
        return "⚠️ 数据库中还没有文档，请先上传并入库。"

    # 2. 拼接上下文
    context = "\n\n".join(documents)

    # 3. 构造提示词，交给大模型
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template("""
你是一个智能助手。请根据以下已知信息回答用户的问题。
如果已知信息不足以回答，请诚实地说不知道，不要编造答案。

已知信息：
{context}

用户问题：{question}

请给出清晰、准确的回答。
""")

    llm = _get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# ============ Gradio 界面 ============
def main():
    import gradio as gr

    # 全局状态
    state = {"vectorstore": None}

    def init_state():
        """初始化向量数据库"""
        if state["vectorstore"] is None:
            emb, db, collection = _get_vectorstore()
            state["vectorstore"] = (emb, db, collection)
        return state["vectorstore"]

    def handle_upload(file):
        """处理文件上传"""
        if file is None:
            return "请上传文件"
        try:
            emb, db, collection = init_state()
            count = _add_to_vectorstore(file.name, emb, collection)
            total = collection.count()
            return f"✅ 已入库 {count} 块，当前共 {total} 条记录"
        except Exception as e:
            return f"❌ 上传失败: {e}"

    def handle_chat(message):
        """处理用户提问"""
        try:
            emb, db, collection = init_state()
            if collection.count() == 0:
                return "⚠️ 请先上传文档并入库。"
            return _ask_rag(message, emb, collection)
        except Exception as e:
            return f"❌ 出错了: {e}"

    def clear_chat():
        return ""

    with gr.Blocks(title="AI 知识库问答系统") as demo:
        gr.Markdown("""
# 📚 AI 知识库问答系统

1. 上传文档（PDF / TXT）
2. 点击「上传并入库」
3. 在下方提问，AI 会根据文档内容回答
        """)

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="📤 上传文档", file_types=[".pdf", ".txt"])
                upload_btn = gr.Button("上传并入库", variant="primary")
                status = gr.Textbox(label="状态", interactive=False)

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="对话", height=400)
                msg_input = gr.Textbox(placeholder="输入你的问题...", label="提问")
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空")

        upload_btn.click(handle_upload, [file_input], [status])
        msg_input.submit(handle_chat, [msg_input], [chatbot])
        submit_btn.click(handle_chat, [msg_input], [chatbot])
        clear_btn.click(clear_chat, outputs=[msg_input])

    demo.launch(server_port=7860)


if __name__ == "__main__":
    main()

