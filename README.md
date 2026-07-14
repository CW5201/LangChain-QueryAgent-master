# SmartKB - 智能知识库与数据分析Agent系统

> RAG + Agent + 企业级知识库解决方案 | AI应用工程师项目

## 项目简介

SmartKB 是一个面向企业的智能知识库问答平台，集成了 **RAG（检索增强生成）** 和 **Agent（智能体）** 两大引擎，打通了"文档上传→解析分块→向量存储→语义检索→智能问答→效果评估"的全链路。

**解决的问题：**
- 企业内部文档分散，查找信息耗时
- 传统LLM不了解企业内部知识，回答产生幻觉
- 缺乏量化评估手段，RAG优化无从下手

## 快速开始

### 环境要求
- Python 3.9+
- Ollama（本地模式）或 云端API Key

### 安装与启动

```bash
# 1. 安装依赖
cd backend
pip install -r requirements.txt

# 2. 启动后端
python -m uvicorn backend.main:app --reload --port 8000

# 3. 启动前端（新终端）
cd frontend
streamlit run app.py
```

### 访问地址

| 服务 | 地址 |
|------|------|
| 前端界面 | http://localhost:8501 |
| API文档 | http://localhost:8000/docs |
| 健康检查 | http://localhost:8000/api/health |

## 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| 后端 | FastAPI | 异步API框架 |
| 前端 | Streamlit | 快速原型 |
| LLM编排 | LangChain | RAG + Agent |
| 向量数据库 | ChromaDB | HNSW余弦相似度检索 |
| 关系型数据库 | SQLite | 业务数据存储 |
| 文档解析 | PyMuPDF | PDF/TXT/MD |
| 本地模型 | Ollama | qwen2.5:7b |
| 云端API | 通义千问/DeepSeek | OpenAI兼容接口 |

## 项目结构

```
backend/
├── main.py              # FastAPI入口
├── config.py            # 配置管理（支持热更新）
├── config.yaml          # 配置文件
├── api/router.py        # API路由（15个端点）
├── core/
│   ├── rag_engine.py    # RAG检索生成引擎
│   ├── agent_engine.py  # Agent智能体 + 防烧钱熔断
│   ├── parser.py        # 多格式文档解析
│   ├── database.py      # SQLite + ChromaDB双库
│   ├── tools.py         # Agent工具（知识库查询+网络搜索）
│   ├── eval_engine.py   # LLM-as-a-Judge评估
│   ├── alert.py         # 飞书/钉钉/企微异步告警
│   └── models_factory.py # 模型工厂（热插拔）
└── models/schemas.py    # 数据模型
```

## 核心功能

### 1. RAG检索增强生成

```
用户提问 → Embedding向量化 → ChromaDB余弦相似度检索Top-K
→ 拼接上下文（带来源标注） → LLM基于上下文生成回答
```

- 分块策略：RecursiveCharacterTextSplitter，中文分隔符（段落→句子→标点→空格）
- 参数：chunk_size=500, overlap=100, top_k=3
- 多轮对话：System Prompt动态追加最近5轮历史

### 2. Agent智能体 + 防烧钱熔断

```
用户提问 → 先走RAG检索
  ├─ 有命中 → 直接RAG生成
  └─ 无命中 → Agent模式（自动调用工具）
       ├─ database_query：知识库查询
       └─ web_search：网络搜索
       监控intermediate_steps工具调用次数
       超过5次 → CircuitBreaker熔断 + 告警
```

### 3. LLM-as-a-Judge评估

```
输入：问题 + 检索上下文 + 生成回答
三维评分（1-5分）：
  - 相关性（Relevance）：回答是否紧扣问题
  - 完整性（Completeness）：是否覆盖所有要点
  - 准确性（Accuracy）：事实是否正确
传统指标：Jaccard相似度 + 关键词命中率
```

### 4. 其他特性

- **模型热插拔**：config.yaml改mode字段即可切换本地/云端
- **配置热更新**：PUT /api/config 无需重启
- **异步告警**：asyncio.create_task不阻塞主流程
- **SSE流式输出**：Agent思考过程可视化（thinking/tool_call/answer事件流）

## API接口（15个端点）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/documents/upload` | POST | 上传文档 |
| `/api/documents` | GET | 文档列表 |
| `/api/documents/{id}` | GET | 文档详情 |
| `/api/documents/{id}` | DELETE | 删除文档 |
| `/api/chat` | POST | 普通问答 |
| `/api/chat/stream` | POST | 流式问答（SSE） |
| `/api/conversations/{id}` | GET | 对话历史 |
| `/api/conversations/{id}` | DELETE | 清空对话 |
| `/api/feedback` | POST | 点赞/点踩 |
| `/api/eval/run` | POST | 运行LLM-as-a-Judge评估 |
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
  cloud:
    model_name: qwen-turbo
    api_key: ""
  local:
    base_url: http://localhost:11434
    model_name: qwen2.5:7b

rag:
  chunk_size: 500
  chunk_overlap: 100
  top_k: 3

agent:
  max_tool_calls: 5    # 熔断阈值

alert:
  feishu_webhook: ""
  dingtalk_webhook: ""
  wechat_webhook: ""
```

## 面试要点

**Q: 项目解决了什么问题？**
> 企业文档分散，员工找信息平均30分钟。SmartKB把文档统一导入知识库，用自然语言提问秒级回答。

**Q: RAG分块策略怎么设计的？**
> 用RecursiveCharacterTextSplitter，自定义中文分隔符序列（段落→句子→标点→空格）递归切分。chunk_size=500平衡精度和完整性，overlap=100保证跨边界语义不丢失。

**Q: 防烧钱熔断怎么实现的？**
> 监控AgentExecutor的intermediate_steps，统计工具调用次数，超过5次主动raise终止执行，同时异步推送飞书/钉钉告警。30秒后半恢复，防止永久熔断。

**Q: 为什么选ChromaDB？**
> 嵌入式部署零运维，适合万级文档的中小规模知识库。如果数据量增长，可迁移到Milvus。

## License

MIT
