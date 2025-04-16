import asyncio
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import settings
from .models.graph_state import GraphState
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
class AsyncEmbedder(OpenAIEmbeddings):
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.embed_documents(texts)
        )

# 初始化组件
embedding = AsyncEmbedder(model=settings.EMBEDDING_MODEL, api_key=settings.API_KEY, base_url=settings.BASE_URL)
vector_db = Chroma(
    persist_directory=settings.CHROMADB_DIR,
    collection_name=settings.CHROMADB_COLLECTION,
    embedding_function=embedding
)
llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.7, api_key=settings.API_KEY, base_url=settings.BASE_URL)

async def retrieve_node(state: GraphState) -> dict:
    """异步检索节点"""
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(
        None,
        lambda: vector_db.similarity_search(state.query, k=3)
    )
    await state.update_docs([{
        "content": doc.page_content,
        "metadata": doc.metadata
    } for doc in docs])
    return {"retrieved_docs": state.retrieved_docs}

async def generate_node(state: GraphState) -> dict:
    """异步生成节点"""
    prompt = ChatPromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    chain = prompt | llm
    response = await chain.ainvoke({
        "context": "\n".join(d["content"] for d in state.retrieved_docs),
        "query": state.query
    })
    await state.update_answer(response.content)
    return {"answer": state.answer}

async def create_vector_db_node(state: GraphState) -> dict:
    """创建/重建ChromaDB向量数据库节点"""
    try:
        # 1. 清理旧数据库
        if os.path.exists(settings.CHROMADB_DIR):
            shutil.rmtree(settings.CHROMADB_DIR)
        
        # 2. 加载PDF文档
        loader = PyPDFLoader(settings.INPUT_PDF)
        documents = loader.load()
        
        # 3. 按语言分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "。", "！", "？"] if settings.TEXT_LANGUAGE == "Chinese" else ["\n\n", ".", "!", "?"]
        )
        splits = text_splitter.split_documents(documents)
        
        # 4. 创建向量库
        vector_db = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=settings.CHROMADB_DIR,
            collection_name=settings.CHROMADB_COLLECTION
        )
        vector_db.persist()
        
        return {"status": "success", "doc_count": len(splits)}
    except Exception as e:
        return {"status": "error", "message": str(e)}