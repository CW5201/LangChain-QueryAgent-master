# FastAPI入口文件

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("SmartKB 启动...")
    yield
    print("SmartKB 关闭")


app = FastAPI(title="SmartKB", version="1.0.0", lifespan=lifespan)

# 允许跨域（前端能访问后端）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "SmartKB API", "docs": "/docs"}
