from fastapi import APIRouter, Depends, HTTPException
from .models.graph_state import GraphState
from .models.schemas import QueryRequest, ResponseModel
from .workflows import get_workflow
from config import settings
router = APIRouter()

@router.post("/chat/completions", response_model=ResponseModel)
async def chat_completions(
    request: QueryRequest,
    workflow=Depends(get_workflow)
):
    try:
        settings.logger.info(request)
        settings.logger.info(f"Received query: {request.text}")
        state = GraphState(query=request.text)
        await workflow.ainvoke(state)
        return {
            "answer": state.answer,
            "docs": state.retrieved_docs
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@router.post('rebuild_db')
async def rebuild_db(workflow=Depends(get_workflow)):
    state = GraphState()
    result = await workflow.ainvoke(state)
    if state.init_status.get("status") == "success":
        return {
            "status": "success",
            "doc_count": state.init_status.get("doc_count")
        }
    else:
        raise HTTPException(500, detail=state.init_status["message"])