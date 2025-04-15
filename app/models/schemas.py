from pydantic import BaseModel
from typing import List, Dict
class QueryRequest(BaseModel):
    text: str

class ResponseModel(BaseModel):
    answer: str
    docs: List[Dict]