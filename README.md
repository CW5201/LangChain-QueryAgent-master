# SmartKB - 智能知识库问答系统

> RAG + Agent，面向企业的AI知识库解决方案

## 项目简介

SmartKB 是一个智能知识库问答平台，你可以上传文档（PDF/TXT/MD），然后用自然语言提问，系统会自动从文档中找到答案。

**核心流程：**
```
用户提问 → 语义检索知识库 → 拼接上下文 → 大模型生成回答
```

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 启动后端

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### 3. 启动前端

```bash
cd frontend
streamlit run app.py
```

### 4. 访问

- 前端：http://localhost:8501
- API文档：http://localhost:8000/docs

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | FastAPI |
| 前端 | Streamlit |
| LLM编排 | LangChain |
| 向量数据库 | ChromaDB |
| 本地模型 | Ollama |
| 文档解析 | PyMuPDF |

## 项目结构

```
backend/
├── main.py              # FastAPI入口
├── config.py            # 配置管理
├── config.yaml          # 配置文件
├── api/router.py        # API路由（15个端点）
├── core/
│   ├── rag_engine.py    # RAG检索生成引擎
│   ├── agent_engine.py  # Agent智能体 + 熔断
│   ├── parser.py        # 文档解析
│   ├── database.py      # 数据库（SQLite+ChromaDB）
│   ├── tools.py         # Agent工具
│   ├── eval_engine.py   # 评估引擎
│   ├── alert.py         # 告警通知
│   └── models_factory.py # 模型工厂
└── models/schemas.py    # 数据模型
```

## API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/documents/upload` | POST | 上传文档 |
| `/api/documents` | GET | 文档列表 |
| `/api/documents/{id}` | GET | 文档详情 |
| `/api/documents/{id}` | DELETE | 删除文档 |
| `/api/chat` | POST | 问答 |
| `/api/chat/stream` | POST | 流式问答 |
| `/api/conversations/{id}` | GET | 对话历史 |
| `/api/conversations/{id}` | DELETE | 清空对话 |
| `/api/feedback` | POST | 反馈 |
| `/api/eval/run` | POST | 运行评估 |
| `/api/eval/metrics` | GET | 评估指标 |
| `/api/config` | GET | 获取配置 |
| `/api/config` | PUT | 热更新配置 |
| `/api/stats` | GET | 系统统计 |
| `/api/health` | GET | 健康检查 |

## 配置

编辑 `backend/config.yaml`：

```yaml
model:
  mode: local          # local=本地Ollama, cloud=云端API
  local:
    base_url: http://localhost:11434
    model_name: qwen2.5:7b
```

## License

MIT
