from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    # "explain"  = normal Q&A (hybrid retrieval + rerank + guardrail + evaluator)
    # "revise"   = broad-coverage revision notes for the whole topic
    # "practice" = broad-coverage practice Q&A for the whole topic
    # revise/practice ignore `question` and require `topic` (which chunks to
    # cover) since they're not answering a specific query.
    mode: Literal["explain", "revise", "practice"] = "explain"
    question: str = Field(default="", min_length=0)
    topic: Optional[str] = None
    debug: bool = False
    # Prior turns for this topic's conversation, oldest first. Only used for
    # mode="explain" — folded into retrieval for follow-up questions and
    # passed to the answer prompt for continuity. Client owns persistence
    # (e.g. localStorage) — the backend is stateless across requests.
    history: Optional[list[ChatTurn]] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    topic: Optional[str] = None
    mode: str = "explain"
    # Only meaningful for mode="explain" — revise/practice skip the
    # guardrail/evaluator agents since they're not a single-query pipeline.
    guardrail_passed: Optional[bool] = None
    evaluator_verdict: Optional[str] = None
    debug_info: Optional[dict] = None


class IngestResponse(BaseModel):
    source: str
    topic: str
    doc_type: str
    chunks_added: int
    reindexed: bool


class BatchIngestResult(BaseModel):
    filename: str
    ok: bool
    source: Optional[str] = None
    doc_type: Optional[str] = None
    chunks_added: Optional[int] = None
    error: Optional[str] = None


class BatchIngestResponse(BaseModel):
    results: list[BatchIngestResult]
    reindexed: bool


class TopicsResponse(BaseModel):
    topics: list[str]


class EvalRunRequest(BaseModel):
    topic: Optional[str] = None
    include_faithfulness: bool = True


class EvalSummary(BaseModel):
    questions_evaluated: int
    avg_keyword_recall: float
    avg_faithfulness: Optional[float] = None
    guardrail_pass_rate: float
    evaluator_pass_rate: float
    avg_latency_s: float