from pydantic import BaseModel, Field

class RAGRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="用户提出的问题"
    )
    candidate_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Hybrid 检索候选数量"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Reranker 最终保留数量"
    )


class RAGSource(BaseModel):
    source: str
    page: int | None = None
    score: float


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[RAGSource]
