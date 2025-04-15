from fastapi import Depends, Request
from .workflows import create_rag_workflow
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

async def get_workflow():
    """获取工作流依赖项"""
    return create_rag_workflow()

def get_vector_db(request: Request) -> Chroma:
    """获取向量数据库实例"""
    return request.app.state.vector_db

def get_llm(request: Request) -> ChatOpenAI:
    """获取LLM实例"""
    return request.app.state.llm