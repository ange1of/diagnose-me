from typing import List

from pydantic import BaseModel


class SymptomList(BaseModel):
    symptoms: List[str]


class SearchSymptomQuery(BaseModel):
    query: str