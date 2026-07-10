"""
SmartKB FastAPI主入口模块

本模块是后端服务的入口点，负责：
1. 加载配置文件
2. 初始化日志系统
3. 初始化数据库和引擎
4. 创建FastAPI应用
5. 配置中间件

启动方式：
┌─────────────────────────────────────────────────────────────────┐
│                        启动命令                                  │
├─────────────────────────────────────────────────────────────────┤
│  开发模式:                                                      │
│    uvicorn backend.main:app --reload --port 8000               │
│                                                                  │
│  生产模式:                                                      │
│    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4 │
│                                                                  │
│  直接运行:                                                      │
│    python -m backend.main                                       │
└─────────────────────────────────────────────────────────────────┘

系统架构：
┌─────────────────────────────────────────────────────────────────┐
│                        启动流程                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. 加载配置 (config.yaml)                                      │
│          ↓                                                      │
│  2. 初始化日志系统                                               │
│          ↓                                                      │
│  3. 初始化示例数据库                                             │
│          ↓                                                      │
│  4. 初始化SQLite + ChromaDB                                     │
│          ↓                                                      │
│  5. 初始化RAG引擎 (LLM + 嵌入模型)                              │
│          ↓                                                      │
│  6. 初始化Agent引擎 (工具调用)                                   │
│          ↓                                                      │
│  7. 注入依赖到路由模块                                           │
│          ↓                                                      │
│  8. FastAPI服务就绪，等待请求                                    │
└─────────────────────────────────────────────────────────────────┘
"""

import os
import yaml
import logging
from dotenv import load_dotenv
from logging.handlers import TimedRotatingFileHandler
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 项目内部模块
from backend.api.router import router, set_dependencies
from backend.core.database import DatabaseManager
from backend.core.rag_engine import RAGEngine
from backend.core.agent_engine import AgentEngine
from backend.core.tools import init_sample_database

# 加载.env环境变量
load_dotenv()


# ============================================================================
# 配置和日志
# ============================================================================

def load_config() -> dict:
    """
    加载配置文件
    
    从 backend/config.yaml 读取配置
    
    Returns:
        配置字典，文件不存在则返回空字典
    """
    config_path = "backend/config.yaml"
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    return {}


def setup_logging(config: dict):
    """
    设置日志系统
    
    配置日志格式、输出目标、切割策略
    
    日志配置：
    - 控制台输出：实时查看
    - 文件输出：按天切割，保留7天
    
    Args:
        config: 系统配置字典
    """
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 文件输出（按天切割）
    file_handler = TimedRotatingFileHandler(
        "logs/smartkb.log",
        when="midnight",
        interval=1,
        backupCount=log_config.get("max_days", 7)
    )
    file_handler.setFormatter(formatter)
    
    # 配置根日志
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler]
    )


# ============================================================================
# 应用生命周期管理
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    在应用启动时初始化资源，关闭时释放资源
    
    启动流程：
    ┌─────────────────────────────────────────────────────────────┐
    │  1. 加载配置 (config.yaml)                                  │
    │          ↓                                                  │
    │  2. 初始化日志系统                                           │
    │          ↓                                                  │
    │  3. 初始化示例数据库（销售数据、用户数据）                   │
    │          ↓                                                  │
    │  4. 初始化SQLite + ChromaDB                                  │
    │          ↓                                                  │
    │  5. 初始化RAG引擎 (LLM + 嵌入模型)                          │
    │          ↓                                                  │
    │  6. 初始化Agent引擎 (工具调用)                               │
    │          ↓                                                  │
    │  7. 注入依赖到路由模块                                       │
    │          ↓                                                  │
    │  8. ✅ 服务就绪                                              │
    └─────────────────────────────────────────────────────────────┘
    """
    # ========== 启动阶段 ==========
    
    # 1. 加载配置
    config = load_config()
    
    # 2. 初始化日志
    setup_logging(config)
    logging.info("=" * 50)
    logging.info("SmartKB 服务启动中...")
    logging.info("=" * 50)
    
    # 3. 初始化示例数据库
    init_sample_database()
    logging.info(" 示例数据库初始化完成")
    
    # 4. 初始化数据库管理器
    db_manager = DatabaseManager(
        sqlite_path=config.get("database", {}).get("sqlite_path", "data/smartkb.db"),
        chroma_path=config.get("database", {}).get("chroma_path", "data/chroma_db")
    )
    logging.info(" SQLite + ChromaDB 初始化完成")
    
    # 5. 初始化RAG引擎
    rag_engine = RAGEngine(db_manager, config)
    logging.info(" RAG引擎初始化完成")
    
    # 6. 初始化Agent引擎
    agent_engine = AgentEngine(rag_engine, db_manager, config)
    logging.info(" Agent引擎初始化完成")
    
    # 7. 注入依赖到路由模块
    set_dependencies(db_manager, rag_engine, agent_engine, config)
    logging.info(" 路由模块初始化完成")
    
    logging.info("=" * 50)
    logging.info("SmartKB 服务启动完成！")
    logging.info(f"API文档: http://localhost:8000/docs")
    logging.info("=" * 50)
    
    yield  # ========== 应用运行期间 ==========
    
    # ========== 关闭阶段 ==========
    logging.info("SmartKB 服务关闭")


# ============================================================================
# 创建FastAPI应用
# ============================================================================

app = FastAPI(
    title="SmartKB - 智能知识库与数据分析Agent系统",
    description="一个面向企业内部的知识管理与智能分析平台，支持RAG问答、Agent工具调用、模型热插拔",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # 允许所有来源
    allow_credentials=True,      # 允许携带凭证
    allow_methods=["*"],         # 允许所有HTTP方法
    allow_headers=["*"],         # 允许所有请求头
)

# 注册API路由
app.include_router(router, prefix="/api")


# ============================================================================
# 根路径接口
# ============================================================================

@app.get("/")
async def root():
    """
    根路径接口
    
    返回API基本信息和文档地址
    """
    return {
        "name": "SmartKB API",
        "version": "1.0.0",
        "docs": "/docs",
        "description": "智能知识库与数据分析Agent系统"
    }


# ============================================================================
# 直接运行入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    config = load_config()
    server_config = config.get("server", {})
    
    uvicorn.run(
        "main:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=True    # 开发模式自动重载
    )
