from langgraph.graph import Graph
from .nodes import retrieve_node, generate_node, create_vector_db_node

def create_rag_workflow():
    workflow = Graph()
    workflow.add_node("init_db", create_vector_db_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_conditional_edges(
        "init_db",
        lambda x : "retrieve" if x["status"] == "success" else "__end__",
    )
    workflow.add_edge("retrieve", "generate")
    workflow.set_entry_point("retrieve")
    workflow.set_finish_point("generate")
    return workflow.compile()

async def get_workflow() -> Graph:
    """获取工作流的异步依赖项"""
    return create_rag_workflow()