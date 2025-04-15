from langgraph.graph import Graph
from .nodes import retrieve_node, generate_node

def create_rag_workflow():
    workflow = Graph()
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge("retrieve", "generate")
    workflow.set_entry_point("retrieve")
    workflow.set_finish_point("generate")
    return workflow.compile()

async def get_workflow() -> Graph:
    """获取工作流的异步依赖项"""
    return create_rag_workflow()