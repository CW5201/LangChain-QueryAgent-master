# SmartKB 学习路线（30天）

> 前提：每天有 2-3 小时，有 Python 基础（会写函数和类就行）
> 目标：学完后能独立看懂并修改这个项目

---

## 第一周：Python 后端基础 + FastAPI

**目标：能独立写一个 REST API 服务**

| 天 | 学什么 | 对应项目文件 | 怎么做 |
|---|--------|-------------|--------|
| Day 1 | Python 复习：列表推导式、装饰器、上下文管理器、`@property` | `backend/models/schemas.py` | 读这个文件，理解 Pydantic 模型是怎么定义的 |
| Day 2 | FastAPI 入门：路由、路径参数、查询参数、请求/响应模型 | `backend/api/router.py` | 对照着看每个 `@router.get/post` 怎么写 |
| Day 3 | 异步编程：`async/await`、同步 vs 异步的区别 | `backend/api/router.py` 中所有 `async def` | 理解为什么 API 接口要用 async |
| Day 4 | Pydantic 深入：字段校验、枚举、Optional、自定义验证器 | `backend/models/schemas.py` | 看 `ModelMode`、`DocumentStatus` 枚举 |
| Day 5 | 中间件、CORS、依赖注入 `Depends` | `backend/main.py`（CORS）、`router.py`（Depends） | 理解跨域是怎么配的，Depends 怎么传数据 |
| Day 6 | **实战**：手写一个 Todo API（增删改查） | 不看项目，自己从零写 | 用 FastAPI + SQLite，写 4 个接口 |
| Day 7 | 复习 + 调试 | 浏览器访问 `http://localhost:8000/docs` | 用 Swagger 文档测试每个接口 |

**关键文件**：`backend/main.py`、`backend/api/router.py`、`backend/models/schemas.py`

---

## 第二周：数据库 + 文档解析

**目标：理解数据怎么存、怎么取、文档怎么处理**

| 天 | 学什么 | 对应项目文件 | 怎么做 |
|---|--------|-------------|--------|
| Day 8 | SQLite 基础：连接、游标、CRUD、参数化查询防注入 | `backend/core/database.py` | 看所有 `cursor.execute` 的写法 |
| Day 9 | 数据库设计：表结构、主键、外键、索引 | `backend/core/database.py` 开头的 DDL | 理解 4 张表的关系 |
| Day 10 | 文件上传处理：`UploadFile`、文件读写、编码检测 | `backend/api/router.py` 的 upload 接口 | 看懂文件怎么存到磁盘的 |
| Day 11 | PDF 解析：PyMuPDF、逐页提取文本、处理空页/纯图片页 | `backend/core/parser.py` 的 `_extract_from_pdf` | 读一遍，理解 PDF 怎么变文本 |
| Day 12 | 文本分块：`RecursiveCharacterTextSplitter`、chunk_size、overlap | `backend/core/parser.py` 的 `split_text` | 理解为什么要把文本切成小块 |
| Day 13 | **实战**：写一个文档上传→解析→存 SQLite 的小流程 | 不看项目，自己写 | 上传一个 PDF，存到数据库 |
| Day 14 | 复习：串起来跑一遍 | `backend/core/parser.py` + `database.py` | 上传一个 PDF，去数据库看结果 |

**关键文件**：`backend/core/parser.py`、`backend/core/database.py`、`backend/core/tools.py`

---

## 第三周：LangChain + RAG + Agent

**目标：这是项目的核心，理解 AI 怎么工作**

| 天 | 学什么 | 对应项目文件 | 怎么做 |
|---|--------|-------------|--------|
| Day 15 | LangChain 基础：`ChatOpenAI`、`MessagesPlaceholder`、Prompt 模板 | `backend/core/rag_engine.py` 的 `generate()` | 理解 LLM 是怎么调用的 |
| Day 16 | LCEL 语法：`prompt \| llm \| parser`，理解管道链 | `backend/core/rag_engine.py` 第 293 行 | 这行代码就是 RAG 的核心 |
| Day 17 | 嵌入模型：OpenAIEmbeddings、文本→向量、余弦相似度 | `backend/core/models_factory.py` 的 `create_embeddings` | 理解"文本变向量"是什么 |
| Day 18 | 向量数据库：ChromaDB、collection、add/query/delete | `backend/core/database.py` 的 `search_similar` | 理解语义搜索的原理 |
| Day 19 | RAG 完整流程：检索→格式化→拼接提示词→生成回答 | `rag_engine.py` 的 `retrieve → format_docs → generate` | 串读这三个方法 |
| Day 20 | Agent：`create_openai_tools_agent`、`AgentExecutor`、工具注册 | `backend/core/agent_engine.py` 的 `_init_agent` | 理解 Agent 怎么"自己决定用什么工具" |
| Day 21 | Agent 工具：`@tool` 装饰器、`database_query`、`web_search` | `backend/core/tools.py` | 理解工具怎么被 Agent 调用的 |

**关键文件**：`backend/core/models_factory.py`、`backend/core/rag_engine.py`、`backend/core/agent_engine.py`、`backend/core/tools.py`

---

## 第四周：高级特性 + 项目实战

**目标：掌握项目的高级功能，能独立扩展**

| 天 | 学什么 | 对应项目文件 | 怎么做 |
|---|--------|-------------|--------|
| Day 22 | SSE 流式输出：`StreamingResponse`、生成器、`yield` | `backend/api/router.py` 的 `chat_stream` | 理解流式输出怎么实现的 |
| Day 23 | 配置管理：YAML 配置、热更新、`reload_config` | `backend/api/router.py` 的 `update_config` | 理解配置改了怎么不重启生效 |
| Day 24 | 告警系统：`asyncio.create_task`、Webhook、飞书/钉钉 | `backend/core/alert.py` | 理解异步告警怎么发的 |
| Day 25 | RAG 评估：`EvalEngine`、LLM-as-a-Judge、关键词命中率 | `backend/core/eval_engine.py` | 理解怎么给 RAG 打分 |
| Day 26 | 前端：Streamlit 基础、页面路由、session_state、API 调用 | `frontend/app.py` | 理解前端怎么调后端的 |
| Day 27 | Docker：`Dockerfile`、`docker-compose.yml`、容器化部署 | 根目录下的 Docker 文件 | 跑一遍 `docker-compose up` |
| Day 28 | 测试：pytest、写单元测试 | `tests/test_basic.py` | 补一个自己的测试用例 |
| Day 29 | **综合实战**：给项目加一个新功能（比如"删除所有文档"接口） | 从 `router.py` 到 `app.py` 全流程 | 自己写，不抄代码 |
| Day 30 | **复盘**：画架构图、写学习总结、准备面试话术 | `README.md` 的面试问答部分 | 对着过一遍，能讲给别人听 |

---

## 最重要的 3 个概念（一定要搞懂）

1. **RAG**：先检索知识库，再让 AI 基于检索结果回答。不是让 AI 凭空编答案。
2. **Agent**：AI 自己决定用什么工具（查数据库？搜网页？），不是人告诉它。
3. **流式输出**：AI 边想边说，不是等全部想好才给结果。用户体验好很多。

## 每天怎么学

1. **先看代码**（30min）— 读上面标注的关键文件
2. **学知识点**（60min）— B站搜对应关键词看教程
3. **动手写**（60min）— 不抄代码，自己敲一遍

## 遇到不懂的名词直接搜

- "FastAPI Depends 依赖注入"
- "LangChain LCEL 管道"
- "ChromaDB 余弦相似度"
- "SSE Server-Sent Events Python"
- "Pydantic 枚举 校验"
- "SQLite 参数化查询"
- "asyncio create_task 异步任务"
- "Streamlit session_state"
