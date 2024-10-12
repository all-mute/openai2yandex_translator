from pydantic import BaseModel, Field
from typing import List, Literal

class CompletionOptions(BaseModel):
    stream: bool
    temperature: float = Field(0.3, ge=0.0, le=1.0)
    maxTokens: int = Field(..., gt=0)

class Message(BaseModel):
    role: Literal['system', 'assistant', 'user']
    text: str

class CompletionRequest(BaseModel):
    modelUri: str
    completionOptions: CompletionOptions
    messages: List[Message]

class AlternativeMessage(BaseModel):
    role: Literal['system', 'assistant', 'user']
    text: str

class Alternative(BaseModel):
    message: AlternativeMessage
    status: Literal[
        'ALTERNATIVE_STATUS_UNSPECIFIED',
        'ALTERNATIVE_STATUS_PARTIAL',
        'ALTERNATIVE_STATUS_TRUNCATED_FINAL',
        'ALTERNATIVE_STATUS_FINAL',
        'ALTERNATIVE_STATUS_CONTENT_FILTER'
    ]

class Usage(BaseModel):
    inputTextTokens: str
    completionTokens: str
    totalTokens: str

class CompletionResponse(BaseModel):
    alternatives: List[Alternative]
    usage: Usage
    modelVersion: str

class TextEmbeddingRequest(BaseModel):
    modelUri: str
    text: str

class TextEmbeddingResponse(BaseModel):
    embedding: List[float]
    numTokens: str
    modelVersion: str