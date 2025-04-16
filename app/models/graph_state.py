from pydantic import BaseModel
from typing import List, Dict, Optional

class GraphState(BaseModel):
    query: str
    retrieved_docs: List[Dict] = []
    answer: Optional[str] = None
    init_status : Optional[Dict] = None

    async def update_docs(self, docs: List[Dict]):
        self.retrieved_docs = docs
    
    async def update_answer(self, answer: str):
        self.answer = answer

    async def update_init_status(self, status: Dict):
        self.init_status = status