from pydantic import BaseModel
from typing import List, Dict, Optional

class GraphState(BaseModel):
    query: str
    retrieved_docs: List[Dict] = []
    answer: Optional[str] = None

    async def update_docs(self, docs: List[Dict]):
        self.retrieved_docs = docs
    
    async def update_answer(self, answer: str):
        self.answer = answer