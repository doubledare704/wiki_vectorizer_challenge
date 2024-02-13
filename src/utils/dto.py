from typing import List

from pydantic.v1 import BaseModel


class SearchByQueryResult(BaseModel):
    id: int
    similarity: float
    vector: list


class SearchResultList(BaseModel):
    data: List[SearchByQueryResult]
