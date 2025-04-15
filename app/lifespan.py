from contextlib import asynccontextmanager
from fastapi import FastAPI
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from config import settings
import asyncio

from rag_service.app.nodes import AsyncEmbedder

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用全生命周期资源"""
    # ----------------------------
    # 启动时初始化
    # ----------------------------
    print("⏳ 初始化资源...")
    
    # 初始化向量数据库连接
    app.state.vector_db = Chroma(
        persist_directory=settings.CHROMADB_DIR,
        collection_name=settings.CHROMADB_COLLECTION,
        embedding_function=AsyncEmbedder(model=settings.EMBEDDING_MODEL)
    )
    
    # 初始化LLM（带自动清理）
    app.state.llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0.7,
        max_retries=3
    )
    
    # ----------------------------
    # 服务运行中 (yield)
    # ----------------------------
    yield
    
    # ----------------------------
    # 关闭时清理
    # ----------------------------
    print("⏳ 清理资源...")
    
    # 关闭向量数据库连接
    if hasattr(app.state, 'vector_db'):
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: app.state.vector_db.client.close()
        )
        print("✅ ChromaDB 连接已关闭")
    
    # 清理LLM连接池
    if hasattr(app.state, 'llm'):
        await app.state.llm.aclose()
        print("✅ OpenAI 连接池已释放")
    
    print("🎉 资源清理完成")