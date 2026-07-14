"""
SmartKB 入口文件
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("SmartKB 启动中...")
    yield
    print("SmartKB 关闭")


app = FastAPI(
    title="SmartKB",
    description="AI知识库问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# 跨域设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "SmartKB API", "docs": "/docs"}
