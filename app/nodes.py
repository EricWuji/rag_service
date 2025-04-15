import asyncio
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import AsyncEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import settings
from models.graph_state import GraphState
from typing import List
class AsyncEmbedder(AsyncEmbeddings):
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.embed_documents(texts)
        )

# 初始化组件
embedding = AsyncEmbedder(model=settings.EMBEDDING_MODEL)
vector_db = Chroma(
    persist_directory=settings.CHROMADB_DIR,
    collection_name=settings.CHROMADB_COLLECTION,
    embedding_function=embedding
)
llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.7)

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