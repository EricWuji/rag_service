from contextlib import asynccontextmanager
from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from config import settings
import asyncio
import os
from .models.graph_state import GraphState
from .nodes import create_vector_db_node

from app.nodes import AsyncEmbedder

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用全生命周期资源"""
    # ----------------------------
    # 启动时初始化
    # ----------------------------
    print("⏳ 初始化资源...")
    
    if not os.path.exists(settings.CHROMADB_DIR):
        print("⚠️ 未检测到向量库，正在初始化...")
        db_node = await create_vector_db_node(GraphState())
        result = await db_node(GraphState())
        if result["status"] != "success":
            raise RuntimeError(f"初始化失败: {result['message']}")
        
    # 初始化向量数据库连接
    app.state.vector_db = Chroma(
        persist_directory=settings.CHROMADB_DIR,
        collection_name=settings.CHROMADB_COLLECTION,
        embedding_function=AsyncEmbedder(model=settings.EMBEDDING_MODEL, base_url=settings.BASE_URL, api_key=settings.API_KEY),
    )
    
    # 初始化LLM（带自动清理）
    app.state.llm = ChatOpenAI(model=settings.OPENAI_MODEL, 
                               temperature=0.7, api_key=settings.API_KEY, base_url=settings.BASE_URL)

    app.state.logger = settings.logger
    
    # ----------------------------
    # 服务运行中 (yield)
    # ----------------------------
    yield
    
    # ----------------------------
    # 关闭时清理
    # ----------------------------
    print("⏳ 清理资源...")
    
    print("🎉 资源清理完成")