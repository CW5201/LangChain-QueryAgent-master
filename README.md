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

## 📖 项目简介

SmartKB 是一个面向企业的智能知识库与数据分析平台，集成了 **RAG（检索增强生成）** 和 **Agent（智能体）** 技术，支持自然语言问答、文档管理、数据分析等功能。

### 核心特性

- 🤖 **智能问答** - 基于RAG的知识库问答，支持多轮对话
- 🔧 **Agent工具** - 自动选择数据库查询、网页搜索等工具
- 📊 **数据分析** - 自然语言查询销售数据、用户信息等
- ⚡ **模型热插拔** - 本地Ollama/云端API无缝切换
- 🔍 **RAG评估** - LLM-as-a-Judge自动评估问答质量
- 🚨 **告警系统** - 飞书/钉钉/企业微信Webhook通知
- 🐳 **Docker部署** - 一键启动，开箱即用

### 技术亮点

1. **Agent思考过程可视化** - 展示任务规划和工具调用过程
2. **防烧钱熔断机制** - 自动检测并终止死循环调用
3. **配置热更新** - 修改配置无需重启服务
4. **SSE流式输出** - 打字机效果提升用户体验
5. **多模型支持** - 通义千问、DeepSeek、Ollama本地模型

---

## 🏗️ 系统架构

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
│                    FastAPI Backend (无状态)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    API Gateway                        │  │
│  │  /documents  /chat  /config  /stats  /health         │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    Core Engine                        │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │  │
│  │  │   RAG   │  │  Agent  │  │ Parser  │  │ Database│ │  │
│  │  │ Engine  │  │ Engine  │  │         │  │ Manager │ │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │  │
│  │  ┌─────────┐  ┌─────────┐                            │  │
│  │  │  Eval   │  │  Alert  │                            │  │
│  │  │ Engine  │  │ Module  │                            │  │
│  │  └─────────┘  └─────────┘                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    存储层                                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ ChromaDB│  │ MySQL   │  │  Files  │  │ Ollama  │       │
│  │(Vectors)│  │(生产)/  │  │ (Docs)  │  │ (Local) │       │
│  │         │  │SQLite   │  │         │  │         │       │
│  │         │  │(本地)   │  │         │  │         │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| **后端框架** | FastAPI | 高性能异步API框架 |
| **前端框架** | Streamlit | 快速原型开发，生产环境可替换为React/Vue |
| **LLM编排** | LangChain + LangGraph | RAG、Agent编排、状态机控制 |
| **向量数据库** | ChromaDB | 嵌入式向量存储，支持持久化 |
| **关系型数据库** | SQLite/MySQL | 元数据、问答记录、用户信息存储 |
| **文档解析** | PyMuPDF + Unstructured | PDF/TXT/Markdown文档解析 |
| **本地模型** | Ollama | 免费、轻量的本地LLM部署方案 |
| **云端API** | 通义千问/DeepSeek | 支持多种大模型提供商 |
| **容器化** | Docker + Docker Compose | 一键部署，跨平台支持 |

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- Ollama（本地模型模式）或 云端API Key
- Docker & Docker Compose（可选，用于容器化部署）

### 方式一：本地开发

#### 1. 克隆项目

```bash
git clone https://github.com/your-username/SmartKB.git
cd SmartKB
```

#### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 安装后端依赖
cd backend
pip install -r requirements.txt

# 安装前端依赖
cd ../frontend
pip install -r requirements.txt
```

#### 3. 配置环境变量

```bash
# 复制环境变量示例
cp .env.example .env

# 编辑.env文件，填入你的API Key
# DASHSCOPE_API_KEY=your_dashscope_key
# DEEPSEEK_API_KEY=your_deepseek_key
```

#### 4. 启动服务

**终端1 - 启动后端：**
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

**终端2 - 启动前端：**
```bash
cd frontend
streamlit run app.py
```

#### 5. 访问应用

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端界面 | http://localhost:8501 | Streamlit Web应用 |
| API文档 | http://localhost:8000/docs | FastAPI自动文档 |
| 健康检查 | http://localhost:8000/api/health | 服务状态检测 |

### 方式二：Docker部署

```bash
# 一键启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## 📁 项目结构

```
SmartKB/
├── backend/                    # 后端服务
│   ├── api/                    # API路由
│   │   └── router.py          # 所有RESTful接口
│   ├── core/                   # 核心引擎
│   │   ├── rag_engine.py      # RAG检索增强生成引擎
│   │   ├── agent_engine.py    # Agent智能体引擎
│   │   ├── parser.py          # 文档解析器
│   │   ├── database.py        # 数据库管理器
│   │   ├── tools.py           # Agent工具定义
│   │   ├── eval_engine.py     # RAG评估引擎
│   │   └── alert.py           # 告警通知模块
│   ├── models/                 # 数据模型
│   │   └── schemas.py         # Pydantic数据模型
│   ├── main.py                # FastAPI应用入口
│   ├── config.yaml            # 配置文件
│   └── requirements.txt       # 依赖清单
├── frontend/                   # 前端界面
│   ├── app.py                 # Streamlit主界面
│   └── requirements.txt       # 依赖清单
├── data/                       # 数据目录
│   ├── uploads/               # 上传的文档
│   ├── chroma_db/             # 向量数据库
│   └── smartkb.db             # SQLite数据库
├── tests/                      # 测试代码
├── logs/                       # 日志文件
├── .env.example                # 环境变量示例
├── docker-compose.yml          # Docker编排文件
├── Dockerfile.backend          # 后端Docker镜像
├── Dockerfile.frontend         # 前端Docker镜像
└── README.md                   # 项目文档
```

---

## 📚 核心概念

### RAG（检索增强生成）

**问题**：传统LLM不知道你的企业内部知识  
**解决**：先检索知识库，再基于检索结果生成回答

```
用户提问 → 向量化 → 检索相似文档 → 拼接上下文 → LLM生成回答
```

### Agent（智能体）

**能力**：LLM + 工具 + 自主决策

```
用户问"查询销售额" → Agent分析 → 调用database_query工具 → 返回结果
```

### 向量检索

**原理**：文本 → 嵌入模型 → 数字向量 → 计算相似度 → 返回最相关文档

---

## 🔌 API接口文档

### 文档管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/documents/upload` | POST | 上传文档 |
| `/api/documents` | GET | 获取文档列表 |
| `/api/documents/{id}` | GET | 获取文档详情 |
| `/api/documents/{id}` | DELETE | 删除文档 |
| `/api/documents/{id}/preview` | GET | 预览文档分块 |

### 智能问答

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 非流式问答 |
| `/api/chat/stream` | POST | 流式问答（SSE） |
| `/api/chat/feedback` | POST | 提交反馈 |
| `/api/chat/conversation/{id}` | DELETE | 清空对话历史 |

### RAG评估

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/eval` | POST | RAG质量评估 |
| `/api/eval/history` | GET | 获取评估历史 |

### 系统管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/config` | GET | 获取配置 |
| `/api/config` | PUT | 更新配置（热更新） |
| `/api/stats` | GET | 获取统计 |
| `/api/health` | GET | 健康检查 |

> 💡 完整API文档请访问：http://localhost:8000/docs

---

## ⚙️ 配置说明

### 模型配置

编辑 `backend/config.yaml`：

```yaml
model:
  # 运行模式: local(本地Ollama) 或 cloud(云端API)
  mode: local
  
  # 云端配置
  cloud:
    provider: dashscope        # dashscope 或 deepseek
    api_key: ""                # 或通过环境变量设置
    model_name: qwen-turbo     # 模型名称
  
  # 本地配置
  local:
    base_url: http://localhost:11434
    model_name: qwen2.5:7b     # Ollama模型
```

### 数据库配置

```yaml
database:
  sqlite_path: data/smartkb.db    # SQLite数据库路径
  chroma_path: data/chroma_db     # 向量数据库路径
```

---

## 🧪 运行测试

```bash
# 运行所有测试
cd tests
pytest test_basic.py -v

# 运行特定测试
pytest test_parser.py -v
```

---

## 🎯 功能演示

### 1. 知识库管理

- 上传PDF/TXT/Markdown文档
- 自动解析和分块
- 向量存储和检索
- 文档预览和管理

### 2. 智能问答

- 基于RAG的知识库问答
- 多轮对话支持
- 思考过程可视化
- 引用来源展示

### 3. Agent工具调用

- 数据库查询（Sales数据、用户信息）
- 网页搜索
- 自动任务规划
- 防死循环熔断保护

### 4. 模型配置

- 本地/云端模式切换
- 温度、Top-P等参数调整
- 配置热更新（无需重启）

### 5. 系统统计

- 问答次数统计
- 响应时间监控
- 用户反馈分析

---

## 🔥 技术亮点

### 1. 防烧钱熔断机制

**问题**：Agent可能陷入工具调用死循环，导致Token费用飙升  
**解决**：
```python
# 超过5次工具调用自动熔断
if tool_call_count > 5:
    alert_circuit_breaker(session_id, tool_call_count)
    return "已触发熔断保护，请简化问题后重试"
```

### 2. 模型热插拔

支持云端API和Ollama本地模型无缝切换，无需重启服务。

### 3. LLM-as-a-Judge评估

使用LLM自动评估RAG系统的回答质量，提供改进建议。

### 4. 多平台告警通知

支持飞书、钉钉、企业微信Webhook，实时推送系统告警。

---

## 🐛 常见问题

### Q1: 启动报错 `ModuleNotFoundError`

**解决**：确保已安装所有依赖
```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### Q2: Ollama连接失败

**解决**：确保Ollama服务正在运行
```bash
# 启动Ollama
ollama serve

# 拉取模型
ollama pull qwen2.5:7b
```

### Q3: 向量检索不到结果

**解决**：
1. 确认文档已成功上传和解析
2. 检查ChromaDB数据目录权限
3. 尝试增加Top-K参数

### Q4: API Key配置无效

**解决**：
1. 检查.env文件是否正确加载
2. 确认API Key格式正确
3. 查看后端日志获取详细错误信息

---

## 📈 未来优化方向

- [ ] 前端重构：使用React/Vue提升并发性能
- [ ] 数据库升级：生产环境切换MySQL/PostgreSQL
- [ ] 向量检索优化：引入Milvus/Qdrant支持亿级向量
- [ ] 文档解析增强：集成OCR处理图片和扫描件
- [ ] 长期记忆：引入Redis缓存会话状态
- [ ] 监控告警：接入Prometheus+Grafana
- [ ] 混合检索：向量检索+关键词检索双路召回
- [ ] 用户权限：支持多租户和RBAC权限管理

---

## 📖 学习资源

| 资源 | 链接 | 说明 |
|------|------|------|
| LangChain文档 | https://python.langchain.com | Agent、RAG核心 |
| FastAPI教程 | https://fastapi.tiangolo.com | API开发 |
| Streamlit文档 | https://docs.streamlit.io | 前端开发 |
| ChromaDB指南 | https://docs.trychroma.com | 向量数据库 |
| Ollama官网 | https://ollama.ai | 本地模型部署 |

---

## 👨‍💻 关于作者

本项目为AI应用开发实习项目，展示了从需求分析到工程落地的完整能力。

- **技术栈**：Python、FastAPI、LangChain、Streamlit、Docker
- **核心能力**：RAG系统开发、Agent编排、向量数据库应用
- **项目亮点**：防烧钱熔断、模型热插拔、配置热更新

---

## 📄 License

MIT License

---

## 🙏 致谢

感谢以下开源项目和技术：

- [LangChain](https://github.com/langchain-ai/langchain)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [Streamlit](https://github.com/streamlit/streamlit)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Ollama](https://github.com/ollama/ollama)

---

<div align="center">
  <sub>Built with ❤️ by SmartKB Team</sub>
</div>
