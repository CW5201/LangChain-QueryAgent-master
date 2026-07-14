# SmartKB - 智能知识库与数据分析Agent系统

<p align="center">
  <strong>RAG + Agent + 企业级知识库解决方案</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangChain-Latest-orange.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/Docker-Supported-lightgrey.svg" alt="Docker">
</p>

## 项目简介

SmartKB 是一个面向企业的智能知识库与数据分析平台，集成了 **RAG（检索增强生成）** 和 **Agent（智能体）** 两大引擎，打通"文档上传→解析分块→向量存储→语义检索→智能问答→效果评估"的全链路。

### 核心特性

- **智能问答** - 基于RAG的知识库问答，支持多轮对话和来源追溯
- **Agent工具调用** - 自动选择知识库查询、网页搜索等工具
- **防烧钱熔断机制** - 监控工具调用次数，超过阈值自动终止，防止API费用失控
- **Agent思考过程可视化** - SSE事件流展示thinking/tool_call/answer全过程
- **LLM-as-a-Judge评估** - 相关性/完整性/准确性三维1-5分自动评分
- **模型热插拔** - 本地Ollama/云端API（通义千问/DeepSeek/GPT）无缝切换
- **配置热更新** - 运行时修改config.yaml无需重启服务
- **异步告警系统** - 飞书/钉钉/企业微信Webhook通知，asyncio不阻塞主流程
- **SSE流式输出** - 打字机效果提升用户体验

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    前端展示层                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Streamlit (原型)  │  React/Vue (生产)               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    API Gateway (15个端点)             │  │
│  │  /documents  /chat  /eval  /config  /stats  /health  │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    Core Engine                        │  │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐             │  │
│  │  │   RAG   │  │  Agent   │  │  Eval   │             │  │
│  │  │ Engine  │  │  Engine  │  │  Engine │             │  │
│  │  └─────────┘  └──────────┘  └─────────┘             │  │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐             │  │
│  │  │ Parser  │  │ Database │  │  Alert  │             │  │
│  │  │         │  │ Manager  │  │ Module  │             │  │
│  │  └─────────┘  └──────────┘  └─────────┘             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    存储层                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ ChromaDB │  │  SQLite  │  │  Files   │  │  Ollama  │   │
│  │(向量索引) │  │(业务数据) │  │ (文档)   │  │ (本地模型)│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| **后端框架** | FastAPI | 高性能异步API框架，支持SSE流式输出 |
| **前端框架** | Streamlit | 快速原型开发，生产环境可替换为React/Vue |
| **LLM编排** | LangChain + LangChain-Classic | RAG管道、Tool Calling Agent |
| **向量数据库** | ChromaDB | HNSW索引，余弦相似度检索 |
| **关系型数据库** | SQLite | 元数据、问答记录、反馈、对话历史 |
| **文档解析** | PyMuPDF | PDF/TXT/Markdown解析，五种编码自动检测 |
| **本地模型** | Ollama | qwen2.5:7b，私有化部署 |
| **云端API** | 通义千问/DeepSeek/GPT | OpenAI兼容接口 |
| **搜索工具** | ddgs | DuckDuckGo搜索，多后端fallback |
| **容器化** | Docker + Docker Compose | 一键部署 |

---

## 快速开始

### 环境要求

- Python 3.9+
- Ollama（本地模式）或 云端API Key
- Docker & Docker Compose（可选）

### 方式一：本地开发

```bash
# 克隆项目
git clone https://github.com/CW5201/LangChain-QueryAgent-master.git
cd LangChain-QueryAgent-master

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
cd backend
pip install -r requirements.txt
cd ../frontend
pip install -r requirements.txt
```

**启动后端（终端1）：**

```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

**启动前端（终端2）：**

```bash
cd frontend
streamlit run app.py
```

**访问地址：**

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端界面 | http://localhost:8501 | Streamlit Web应用 |
| API文档 | http://localhost:8000/docs | FastAPI Swagger文档 |
| 健康检查 | http://localhost:8000/api/health | 服务状态 |

### 方式二：Docker部署

```bash
docker-compose up -d
docker-compose logs -f    # 查看日志
docker-compose down       # 停止服务
```

---

## 项目结构

```
SmartKB/
├── backend/                    # 后端服务
│   ├── api/
│   │   └── router.py          # 15个RESTful端点
│   ├── core/
│   │   ├── rag_engine.py      # RAG检索增强生成引擎
│   │   ├── agent_engine.py    # Agent智能体 + 防烧钱熔断
│   │   ├── parser.py          # 多格式文档解析器
│   │   ├── database.py        # SQLite + ChromaDB管理
│   │   ├── models_factory.py  # 模型工厂（热插拔）
│   │   ├── tools.py           # Agent工具定义
│   │   ├── eval_engine.py     # LLM-as-a-Judge评估引擎
│   │   └── alert.py           # 异步告警通知
│   ├── models/
│   │   └── schemas.py         # Pydantic数据模型
│   ├── main.py                # FastAPI应用入口
│   ├── config.py              # 配置管理（热更新）
│   ├── config.yaml            # 配置文件
│   └── requirements.txt       # 后端依赖
├── frontend/
│   ├── app.py                 # Streamlit前端（5个页面）
│   └── requirements.txt       # 前端依赖
├── data/                      # 数据目录
│   ├── uploads/               # 上传的文档
│   ├── chroma_db/             # ChromaDB向量数据
│   └── smartkb.db             # SQLite数据库
├── tests/
│   └── test_basic.py          # 单元测试
├── docker-compose.yml         # Docker编排
├── Dockerfile.backend         # 后端镜像
├── Dockerfile.frontend        # 前端镜像
├── .env.example               # 环境变量示例
└── README.md                  # 项目文档
```

---

## 核心概念

### RAG（检索增强生成）

```
用户提问 → Embedding向量化 → ChromaDB余弦相似度检索 → Top-K召回
→ 拼接上下文（带来源标注） → LLM基于上下文生成回答
```

- 分块策略：RecursiveCharacterTextSplitter，中文分隔符序列（段落→句子→标点→空格）
- 参数：chunk_size=500, overlap=100, top_k=3

### Agent（智能体）+ 防烧钱熔断

```
用户提问 → 先走RAG检索
  ├─ 有命中 → 直接RAG生成答案
  └─ 无命中 → Agent模式
       ├─ 分析问题 → 调用工具(database_query/web_search)
       ├─ 监控intermediate_steps工具调用次数
       └─ 超过5次 → CircuitBreaker熔断 + 飞书/钉钉告警
```

### LLM-as-a-Judge评估

```
输入：问题 + 检索上下文 + 生成回答
输出：三维评分（1-5分）
  - 相关性（Relevance）：回答是否紧扣问题
  - 完整性（Completeness）：是否覆盖所有要点
  - 准确性（Accuracy）：事实是否正确
```

---

## API接口文档（15个端点）

### 文档管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/documents/upload` | POST | 上传文档（PDF/TXT/MD） |
| `/api/documents` | GET | 文档列表 |
| `/api/documents/{doc_id}` | GET | 文档详情 |
| `/api/documents/{doc_id}` | DELETE | 删除文档及向量 |

### 智能问答

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 普通问答 |
| `/api/chat/stream` | POST | 流式问答（SSE事件流） |
| `/api/conversations/{conv_id}` | GET | 对话历史 |
| `/api/conversations/{conv_id}` | DELETE | 清空对话 |

### 反馈

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/feedback` | POST | 点赞/点踩反馈 |

### RAG评估

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/eval/run` | POST | 运行LLM-as-a-Judge评估 |
| `/api/eval/metrics` | GET | 获取评估指标 |

### 配置管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/config` | GET | 获取当前配置 |
| `/api/config` | PUT | 热更新配置（无需重启） |

### 系统监控

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/stats` | GET | 系统统计 |
| `/api/health` | GET | 健康检查 |

> 完整API文档：http://localhost:8000/docs

---

## 配置说明

编辑 `backend/config.yaml`：

```yaml
model:
  mode: local                    # local(本地Ollama) 或 cloud(云端API)
  cloud:
    model_name: qwen-turbo
    api_key: ""                  # 或设置环境变量
  local:
    base_url: http://localhost:11434
    model_name: qwen2.5:7b

rag:
  chunk_size: 500                # 分块大小
  chunk_overlap: 100             # 分块重叠
  top_k: 3                       # 检索数量

agent:
  max_tool_calls: 5              # 熔断阈值

alert:
  feishu_webhook: ""             # 飞书机器人
  dingtalk_webhook: ""           # 钉钉机器人
  wechat_webhook: ""             # 企业微信机器人
```

配置热更新：修改config.yaml后调用 `PUT /api/config`，无需重启服务。

---

## 技术亮点（面试重点）

### 1. 防烧钱熔断机制

```python
class CircuitBreaker:
    """监控Agent中间步骤工具调用次数，超过阈值自动熔断"""
    def __init__(self, max_failures=3, reset_timeout=30):
        self.max_failures = max_failures  # 连续失败阈值
        self.state = "closed"             # closed/open/half_open
```

- 监控 `AgentExecutor.intermediate_steps` 中的工具调用
- 超过5次自动终止 + 异步推送飞书/钉钉/企微告警
- 30秒后半恢复，防止永久熔断

### 2. Agent思考过程SSE可视化

```python
# 事件类型：thinking / tool_call / tool_result / answer / error / done
yield f"data: {json.dumps({'type': 'thinking', 'data': '正在检索知识库...'})}\n\n"
```

### 3. LLM-as-a-Judge + Jaccard双评估

```python
# LLM主观评分：相关性/完整性/准确性 1-5分
judge_result = judge_llm.invoke({"question": q, "context": c, "answer": a})

# 传统客观指标：Jaccard相似度 + 关键词命中率
jaccard = len(set1 & set2) / len(set1 | set2)
```

### 4. 多模型热插拔

```python
# models_factory.py 统一工厂层
# 只需修改config.yaml的mode字段，即可切换:
#   local → ChatOllama (qwen2.5:7b)
#   cloud → ChatOpenAI (通义千问/DeepSeek/GPT)
```

### 5. 异步告警不阻塞主流程

```python
# asyncio.create_task() 异步发送
def send_async(self, title, content, level="info"):
    asyncio.create_task(self._send_all_async(title, content, level))
```

---

## 常见问题

### Q1: 启动报错 ModuleNotFoundError

```bash
pip install -r backend/requirements.txt
```

### Q2: Ollama连接失败

```bash
ollama serve              # 启动服务
ollama pull qwen2.5:7b    # 拉取模型
```

### Q3: 向量检索不到结果

1. 确认文档已上传且状态为success
2. 检查 data/chroma_db 目录权限
3. 尝试增大 top_k 参数

### Q4: git push失败 (Failed to connect to github.com)

需要VPN/代理访问GitHub：
```bash
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
git push origin main
```

---

## 未来优化

- 前端重构：React/Vue提升并发性能
- 数据库升级：生产环境切换MySQL/PostgreSQL
- 向量检索：Milvus/Qdrant支持亿级向量
- OCR增强：处理图片和扫描件
- 混合检索：向量检索+关键词检索双路召回
- 用户权限：多租户RBAC权限管理

---

## License

MIT License
