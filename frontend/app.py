"""
AI 知识库问答系统前端界面

基于 Streamlit 构建的 Web 界面。

页面结构：
┌─────────────────────────────────────────────────────────────────┐
│                         SmartKB 前端                             │
├─────────────────────────────────────────────────────────────────┤
│  侧边栏                    │           主内容区                  │
│  ├─ Logo + 标题            │                                    │
│  ├─ 导航菜单               │  根据导航选择显示对应页面：         │
│  │   ├─ 📚 知识库          │  ├─ 知识库管理                     │
│  │   ├─ 💬 智能问答        │  ├─ 智能问答                       │
│  │   ├─ 🧪 RAG评估        │  ├─ RAG评估                        │
│  │   ├─ ⚙️ 模型配置       │  ├─ 模型配置                       │
│  │   └─ 📊 统计面板        │  └─ 统计面板                       │
│  └─ 系统状态               │                                    │
└─────────────────────────────────────────────────────────────────┘

技术特点：
- 纯 Python 实现，无需前端开发经验
- @st.cache_resource 缓存优化
- 响应式布局
- 支持实时交互
"""

import streamlit as st
import requests
import os
import pandas as pd
from datetime import datetime


# ============================================================================
# 常量配置
# ============================================================================

# 后端API基础地址（支持环境变量覆盖）
API_BASE_URL = os.getenv("SMARTKB_API_URL", "http://localhost:8000/api")

# 支持的文件格式
SUPPORTED_FORMATS = ["pdf", "txt", "md"]

# API 端点路径常量
CHAT_ENDPOINT = "/chat"
CHAT_STREAM_ENDPOINT = "/chat/stream"
DOCUMENTS_ENDPOINT = "/documents"
STATS_ENDPOINT = "/stats"
CONFIG_ENDPOINT = "/config"
EVAL_ENDPOINT = "/eval"
HEALTH_ENDPOINT = "/health"
FEEDBACK_ENDPOINT = "/feedback"
CONVERSATIONS_ENDPOINT = "/conversations"
UPLOAD_ENDPOINT = "/documents/upload"


# ============================================================================
# 缓存优化
# ============================================================================

@st.cache_resource
def get_api_session():
    """
    缓存HTTP会话对象

    使用 @st.cache_resource 确保整个应用生命周期内只创建一次
    避免每次用户交互都重新创建连接
    """
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


@st.cache_resource
def get_session_state():
    """缓存默认会话状态"""
    return {
        "conversation_history": [],
        "document_count": 0,
        "chat_count": 0
    }


# ============================================================================
# 页面配置和样式
# ============================================================================

# Streamlit页面配置
st.set_page_config(
    page_title="SmartKB - 智能知识库",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 全局主题 */
    .main > div { padding-top: 2rem; }
    h1, h2, h3 { font-weight: 700; letter-spacing: -0.02em; }

    /* 侧边栏美化 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8; }
    section[data-testid="stSidebar"] .stRadio label { color: #e2e8f0; }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-bottom-color: #334155 !important;
    }

    /* 导航菜单图标加大 */
    .stRadio > label div[data-baseweb="radio"] { font-size: 1.1rem; }

    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
    }

    /* 思考过程块 */
    .thinking-block {
        background: linear-gradient(90deg, #eff6ff 0%, #f0f9ff 100%);
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #1e40af;
    }

    /* 来源块 */
    .source-block {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }

    /* 按钮圆角 */
    .stButton > button { border-radius: 8px; font-weight: 600; }

    /* 隐藏不必要的分隔线 */
    hr { margin: 0.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API调用工具函数
# ============================================================================

def call_api(endpoint: str, method: str = "GET", data: dict = None, files: dict = None):
    """
    调用后端API的统一方法

    封装了所有与后端的通信逻辑，包括：
    - 连接错误处理
    - 超时设置
    - 响应解析

    Args:
        endpoint: API端点路径（如 "/documents"）
        method: HTTP方法（GET/POST/PUT/DELETE）
        data: 请求体数据（JSON格式）
        files: 上传的文件（用于文件上传）

    Returns:
        API响应的JSON数据（字典），失败时返回 None。
        统一格式：{"code": 200, "message": "xxx", "data": {...}}
    """
    url = f"{API_BASE_URL}{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, timeout=60)
        elif method == "POST":
            if files:
                # 文件上传需要特殊处理（multipart/form-data）
                response = requests.post(url, files=files, timeout=120)
            else:
                response = requests.post(url, json=data, timeout=120)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            st.warning(f"不支持的HTTP方法: {method}")
            return None

        # 检查HTTP状态码
        if response.status_code != 200:
            error_detail = response.json().get("detail", "未知错误")
            st.error(f"请求失败 ({response.status_code}): {error_detail}")
            return None

        return response.json()

    except requests.exceptions.ConnectionError:
        st.error("无法连接到后端服务，请确保后端已启动")
        return None
    except requests.exceptions.Timeout:
        st.error("请求超时，请稍后重试")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP错误: {e}")
        return None
    except Exception as e:
        st.error(f"API调用失败: {str(e)}")
        return None


# ============================================================================
# 侧边栏导航
# ============================================================================

def sidebar_navigation():
    """渲染侧边栏导航"""
    with st.sidebar:
        # 品牌标识（使用emoji代替外部图片，避免网络依赖）
        st.markdown("### 📚 SmartKB")
        st.markdown("---")

        # 页面导航菜单
        page = st.radio(
            "导航",
            ["📚 知识库", "💬 智能问答", "🧪 RAG评估", "⚙️ 模型配置", "📊 统计面板"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # 系统状态显示
        st.caption("系统状态")
        health = call_api(HEALTH_ENDPOINT)
        if health and health.get("code") == 200:
            st.success("● 服务运行中")
        else:
            st.error("● 服务未连接")

        return page


# ============================================================================
# 知识库管理页面
# ============================================================================

def page_knowledge_base():
    """
    知识库管理页面

    功能：
    - 上传文档（PDF/TXT/Markdown）
    - 查看文档列表和解析状态
    - 预览文档分块内容
    - 删除文档

    页面布局：
    ┌─────────────────────────────────────────┐
    │  📚 知识库管理                           │
    ├─────────────────────────────────────────┤
    │  [上传文档区域]                          │
    │  ├─ 文件选择器                          │
    │  ├─ 文件信息显示                        │
    │  └─ 上传按钮                            │
    ├─────────────────────────────────────────┤
    │  [文档列表]                              │
    │  ├─ 文档名称 | 状态 | 分块数 | 操作     │
    │  └─ 预览面板（可展开）                  │
    └─────────────────────────────────────────┘
    """
    st.header("📚 知识库管理")

    # ========== 文件上传区域 ==========
    st.subheader("上传文档")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "选择文件",
            type=SUPPORTED_FORMATS,
            help="支持PDF、TXT、Markdown格式，单个文件不超过10MB"
        )

    with col2:
        # 显示文件信息
        if uploaded_file:
            st.info(f"文件: {uploaded_file.name}")
            st.info(f"大小: {uploaded_file.size / 1024:.1f} KB")

    # 上传按钮和处理
    if uploaded_file and st.button("上传并解析", type="primary"):
        with st.spinner("正在上传和解析文档..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            result = call_api(UPLOAD_ENDPOINT, method="POST", files=files)

            # 统一使用后端返回的 {code, message, data} 格式
            if result and result.get("code") == 200:
                chunk_count = result["data"]["chunk_count"]
                st.success(f"上传成功！文档已解析为 {chunk_count} 个分块")
                st.rerun()    # 刷新页面显示新文档
            else:
                error_msg = result.get("message", "上传失败") if result else "上传失败"
                st.error(error_msg)

    st.markdown("---")

    # ========== 文档列表 ==========
    st.subheader("文档列表")

    result = call_api(DOCUMENTS_ENDPOINT)

    # 统一使用 {code, data} 格式
    if result and result.get("code") == 200:
        documents = result["data"]["documents"]

        if not documents:
            st.info("暂无文档，请上传文档")
        else:
            for doc in documents:
                _render_document_card(doc)
    else:
        st.error("无法获取文档列表")


def _render_document_card(doc: dict):
    """
    渲染单个文档卡片

    Args:
        doc: 文档信息字典
    """
    with st.container():
        # 文档信息行
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

        with col1:
            st.write(f"**{doc['filename']}**")

        with col2:
            # 状态徽章
            status = doc.get("status", "unknown")
            status_map = {
                "success": ("✓ 已完成", "success"),
                "processing": ("⏳ 处理中", "warning"),
                "failed": ("✗ 失败", "error")
            }
            if status in status_map:
                text, stype = status_map[status]
                getattr(st, stype)(text)
            else:
                st.info(status)

        with col3:
            st.caption(f"{doc.get('chunk_count', 0)} 个分块")

        with col4:
            if st.button("删除", key=f"del_{doc['id']}"):
                delete_result = call_api(f"{DOCUMENTS_ENDPOINT}/{doc['id']}", method="DELETE")
                if delete_result and delete_result.get("code") == 200:
                    st.success(delete_result.get("message", "删除成功"))
                    st.rerun()

        # 文档预览（可折叠）
        with st.expander(f"预览: {doc['filename']}"):
            preview_result = call_api(f"{DOCUMENTS_ENDPOINT}/{doc['id']}/preview")
            if preview_result and preview_result.get("code") == 200:
                chunks = preview_result["data"].get("chunks", [])
                for i, chunk in enumerate(chunks[:3], 1):
                    st.markdown(f"**分块 {i}:**")
                    st.text_area(f"分块{i}", chunk, height=100, key=f"preview_{doc['id']}_{i}", disabled=True, label_visibility="collapsed")
            else:
                st.info("无法获取预览")

        st.markdown("---")


# ============================================================================
# 智能问答页面
# ============================================================================

def page_chat():
    """
    智能问答页面

    功能：
    - 多轮对话
    - 思考过程可视化
    - 引用来源展示
    - Agent模式切换

    交互流程：
    ┌─────────────┐
    │  用户输入    │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  调用API    │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  显示回答    │
    │  ├─ 正文    │
    │  ├─ 思考过程 │
    │  └─ 引用来源 │
    └─────────────┘
    """
    st.header("💬 智能问答")

    # 初始化会话状态
    _init_chat_state()

    # 侧边栏控制面板
    _render_chat_sidebar()

    # 显示对话历史
    _render_chat_history()

    # 用户输入处理
    _handle_user_input()


def _init_chat_state():
    """初始化聊天会话状态"""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "use_agent" not in st.session_state:
        st.session_state.use_agent = True


def _render_chat_sidebar():
    """渲染聊天页面的侧边栏"""
    with st.sidebar:
        st.markdown("### 问答设置")

        # Agent模式开关
        st.session_state.use_agent = st.checkbox(
            "启用Agent模式",
            value=st.session_state.use_agent,
            help="开启后可使用数据库查询和网页搜索功能"
        )

        # 新建对话按钮
        if st.button("新建对话", type="primary"):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()


def _render_chat_history():
    """渲染对话历史"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 思考过程（可折叠）
            if "thinking" in msg and msg["thinking"]:
                with st.expander("思考过程", expanded=False):
                    for step in msg["thinking"]:
                        st.markdown(f"- {step}")

            # 引用来源（可折叠）
            if "sources" in msg and msg["sources"]:
                with st.expander("引用来源", expanded=False):
                    for source in msg["sources"]:
                        _render_source_block(source)


def _render_source_block(source: dict):
    """渲染引用来源块"""
    filename = source.get("filename", "未知文档")
    content = source.get("content", "")
    st.markdown(f"**📄 {filename}**")
    st.caption(content[:200])


def _handle_user_input():
    """处理用户输入"""
    if user_question := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(user_question)

        # 保存到消息历史
        st.session_state.messages.append({"role": "user", "content": user_question})

        # 调用后端API获取回答
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            with st.spinner("正在思考..."):
                api_response = call_api(
                    CHAT_ENDPOINT,
                    method="POST",
                    data={
                        "question": user_question,
                        "conversation_id": st.session_state.conversation_id,
                        "use_agent": st.session_state.use_agent,
                    },
                )

            if api_response and api_response.get("code") == 200:
                _handle_chat_success(api_response["data"], message_placeholder)
            else:
                _handle_chat_error(api_response, message_placeholder)


def _handle_chat_success(data: dict, message_placeholder):
    """处理问答成功响应"""
    # 更新对话ID
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = data.get("conversation_id")

    answer = data.get("answer", "")
    thinking = data.get("thinking_process", [])
    sources = data.get("sources", [])

    # 显示回答
    message_placeholder.markdown(answer)

    # 显示思考过程
    if thinking:
        with st.expander("思考过程", expanded=False):
            for step in thinking:
                st.markdown(f"- {step}")

    # 显示引用来源
    if sources:
        with st.expander("引用来源", expanded=False):
            for source in sources:
                _render_source_block(source)

    # 保存助手消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "thinking": thinking,
        "sources": sources
    })


def _handle_chat_error(result, message_placeholder):
    """处理问答错误"""
    error_msg = result.get("message", "无法获取回答") if result else "无法连接到服务"
    message_placeholder.error(error_msg)
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"错误: {error_msg}"
    })


# ============================================================================
# 模型配置页面
# ============================================================================

def page_config():
    """
    模型配置页面

    功能：
    - 切换本地/云端模式
    - 配置API Key
    - 调整LLM参数（温度、Top-P等）
    - 调整RAG参数（Top-K、分块大小）

    配置项：
    ┌─────────────────────────────────────────┐
    │  模型模式: 本地 / 云端                   │
    ├─────────────────────────────────────────┤
    │  云端配置:                               │
    │  ├─ 云服务商 (通义千问/DeepSeek)        │
    │  ├─ 模型名称                            │
    │  └─ API Key                             │
    ├─────────────────────────────────────────┤
    │  本地配置:                               │
    │  ├─ Ollama地址                          │
    │  └─ 模型名称                            │
    ├─────────────────────────────────────────┤
    │  LLM参数:                               │
    │  ├─ 温度 (Temperature)                  │
    │  ├─ Top-P                               │
    │  └─ 最大Token数                         │
    ├─────────────────────────────────────────┤
    │  RAG参数:                               │
    │  ├─ 检索Top-K                           │
    │  └─ 分块大小                            │
    └─────────────────────────────────────────┘
    """
    st.header("⚙️ 模型配置")

    # 获取当前配置
    result = call_api(CONFIG_ENDPOINT)

    if not result or result.get("code") != 200:
        st.error("无法获取配置")
        return

    config = result["data"]

    # ========== 模型模式切换 ==========
    st.subheader("模型模式")
    current_mode = config.get("model", {}).get("mode", "local")
    mode = st.radio(
        "选择模型模式",
        ["local", "cloud"],
        index=0 if current_mode == "local" else 1,
        format_func=lambda x: "🏠 本地模式 (Ollama)" if x == "local" else "☁️ 云端模式"
    )

    st.markdown("---")

    # ========== 云端模型配置 ==========
    st.subheader("云端模型配置")
    cloud_config = config.get("cloud_model", {})

    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox(
            "云服务商",
            ["dashscope", "deepseek"],
            index=0 if cloud_config.get("provider") == "dashscope" else 1,
            format_func=lambda x: "通义千问" if x == "dashscope" else "DeepSeek"
        )

    with col2:
        cloud_model = st.text_input("模型名称", value=cloud_config.get("model", "qwen-turbo"))

    api_key = st.text_input(
        "API Key",
        value="",
        type="password",
        help="也可通过环境变量设置: DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY"
    )

    st.markdown("---")

    # ========== 本地模型配置 ==========
    st.subheader("本地模型配置 (Ollama)")
    local_config = config.get("local_model", {})

    col1, col2 = st.columns(2)
    with col1:
        local_url = st.text_input("Ollama地址", value=local_config.get("base_url", "http://localhost:11434"))

    with col2:
        local_model = st.text_input("模型名称", value=local_config.get("model", "qwen2.5:7b"))

    st.markdown("---")

    # ========== LLM参数 ==========
    st.subheader("LLM参数")
    llm_config = config.get("llm", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider(
            "温度 (Temperature)", 0.0, 2.0,
            llm_config.get("temperature", 0.7), 0.1,
            help="值越高回答越随机，值越低回答越确定"
        )
    with col2:
        top_p = st.slider(
            "Top-P", 0.0, 1.0,
            llm_config.get("top_p", 0.9), 0.1,
            help="核采样参数"
        )
    with col3:
        max_tokens = st.number_input(
            "最大Token数", 100, 4096,
            llm_config.get("max_tokens", 2048)
        )

    st.markdown("---")

    # ========== RAG参数 ==========
    st.subheader("RAG配置")
    rag_config = config.get("rag", {})

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider(
            "检索Top-K", 1, 10,
            rag_config.get("top_k", 3),
            help="检索最相关的K个文档片段"
        )
    with col2:
        chunk_size = st.number_input(
            "分块大小", 100, 2000,
            rag_config.get("chunk_size", 500)
        )

    # ========== 应用配置按钮 ==========
    if st.button("应用配置", type="primary", use_container_width=True):
        _apply_config({
            "mode": mode,
            "cloud_provider": provider,
            "cloud_api_key": api_key,
            "cloud_model_name": cloud_model,
            "local_base_url": local_url,
            "local_model_name": local_model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "top_k": top_k
        })


def _apply_config(update_data: dict):
    """应用配置更新"""
    with st.spinner("正在更新配置..."):
        result = call_api(CONFIG_ENDPOINT, method="PUT", data=update_data)

        if result and result.get("code") == 200:
            st.success(result.get("message", "配置更新成功！"))
        else:
            error_msg = result.get("message", "配置更新失败") if result else "配置更新失败"
            st.error(error_msg)


# ============================================================================
# 统计面板页面
# ============================================================================

def page_stats():
    """
    统计面板页面

    显示系统运行统计指标：
    - 总文档数
    - 总问答次数
    - 今日问答数
    - 平均响应时间
    - 点赞率
    """
    st.header("📊 统计面板")

    # 获取统计数据（只调用一次API）
    stats_result = call_api(STATS_ENDPOINT)
    if not stats_result or stats_result.get("code") != 200:
        st.error("无法获取统计数据")
        return

    stats = stats_result["data"]

    # 指标卡片（5列布局）
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("总文档数", stats.get("total_documents", 0))

    with col2:
        st.metric("总问答数", stats.get("total_qa_count", 0))

    with col3:
        st.metric("今日问答", stats.get("today_qa_count", 0))

    with col4:
        st.metric("平均响应时间", f"{stats.get('avg_response_time', 0):.1f}s")

    with col5:
        like_rate = stats.get("like_rate", 0) * 100
        st.metric("点赞率", f"{like_rate:.1f}%")

    st.markdown("---")

    # 图表区域
    st.subheader("问答趋势")

    try:
        # 用随机波动模拟趋势（实际应从后端获取每日数据）
        base_count = stats.get("today_qa_count", 0)
        days = [f"第{7-i}天" for i in range(7)]
        values = [max(1, base_count // 7 + (i - 3) * 2) for i in range(7)]
        chart_df = pd.DataFrame({"日期": days, "问答数": values}).set_index("日期")
        st.line_chart(chart_df)
    except Exception:
        st.info("无法加载趋势数据")


# ============================================================================
# RAG评估页面
# ============================================================================

def page_eval():
    """
    RAG评估页面

    功能：
    - 单条问答质量评估（LLM-as-a-Judge）
    - 查看历史评估结果
    - 展示评估分数和改进建议

    评估维度：
    - 上下文相关性
    - 忠实度
    - 答案相关性
    """
    st.header("🧪 RAG评估")
    st.info("基于LLM-as-a-Judge方法，自动评估RAG系统的检索和生成质量")

    # ========== 单条评估 ==========
    st.subheader("单条评估")

    with st.form("eval_form"):
        col1, col2 = st.columns(2)

        with col1:
            question = st.text_area("问题", height=100, placeholder="输入评估的问题...")

        with col2:
            answer = st.text_area("回答", height=100, placeholder="输入待评估的回答...")

        context = st.text_area(
            "检索到的上下文", height=150,
            placeholder="输入检索到的相关文档片段..."
        )

        submitted = st.form_submit_button("开始评估", type="primary")

    if submitted and question and answer:
        _run_evaluation(question, answer, context)

    # ========== 历史评估记录 ==========
    st.markdown("---")
    st.subheader("历史评估记录")

    history = call_api(f"{EVAL_ENDPOINT}/metrics")
    if history and history.get("code") == 200:
        records = history["data"]
        st.write(f"**评估时间**: {records.get('timestamp', 'N/A')}")
        st.write(f"**问答统计**: {records.get('qa_stats', {})}")
    else:
        st.info("暂无历史评估记录")


def _run_evaluation(question: str, answer: str, context: str):
    """运行单条评估"""
    with st.spinner("正在评估中..."):
        result = call_api(EVAL_ENDPOINT, method="POST", data={
            "question": question,
            "answer": answer,
            "context": context
        })

    if result:
        st.success("评估完成！")
        _render_evaluation_scores(result)


def _render_evaluation_scores(result: dict):
    """渲染评估分数"""
    scores = result.get("data", {}).get("scores", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        ctx_score = scores.get("context_relevance", 0)
        st.metric("上下文相关性", f"{ctx_score:.2f}",
                  delta="好" if ctx_score > 0.7 else "需改进")

    with col2:
        faith_score = scores.get("faithfulness", 0)
        st.metric("忠实度", f"{faith_score:.2f}",
                  delta="好" if faith_score > 0.7 else "需改进")

    with col3:
        ans_score = scores.get("answer_relevance", 0)
        st.metric("答案相关性", f"{ans_score:.2f}",
                  delta="好" if ans_score > 0.7 else "需改进")

    # 综合评分进度条
    overall = scores.get("overall", 0)
    st.progress(overall)
    st.write(f"**综合评分**: {overall:.2f}")

    # 改进建议
    suggestions = result.get("data", {}).get("suggestions", [])
    if suggestions:
        st.subheader("改进建议")
        for i, suggestion in enumerate(suggestions, 1):
            st.write(f"{i}. {suggestion}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数

    控制页面路由和显示
    """
    page = sidebar_navigation()

    # 页面路由
    page_map = {
        "📚 知识库": page_knowledge_base,
        "💬 智能问答": page_chat,
        "🧪 RAG评估": page_eval,
        "⚙️ 模型配置": page_config,
        "📊 统计面板": page_stats,
    }

    if page in page_map:
        page_map[page]()


if __name__ == "__main__":
    main()