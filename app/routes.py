from fastapi import APIRouter, Depends, HTTPException
from .models.graph_state import GraphState
from .models.schemas import QueryRequest, ResponseModel
from workflows import get_workflow

router = APIRouter()

@router.post("/chat/completions", response_model=ResponseModel)
async def chat_completions(
    request: QueryRequest,
    workflow=Depends(get_workflow)
):
    try:
        state = GraphState(query=request.text)
        await workflow.ainvoke(state)
        return {
            "answer": state.answer,
            "docs": state.retrieved_docs
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))