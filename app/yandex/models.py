from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field, model_validator
from enum import Enum

# Request Models
class CompletionOptions(BaseModel):
    stream: Optional[bool]
    temperature: Optional[float] = Field(default=0.3, ge=0, le=1.0)
    maxTokens: Optional[str]

class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolCall(BaseModel):
    functionCall: FunctionCall

class ToolCallList(BaseModel):
    toolCalls: List[ToolCall]

class FunctionResult(BaseModel):
    name: str
    content: str

class ToolResult(BaseModel):
    functionResult: FunctionResult

class ToolResultList(BaseModel):
    toolResults: List[ToolResult]

class Message(BaseModel):
    role: Literal['system', 'assistant', 'user']
    text: Optional[str] = None
    toolCallList: Optional[ToolCallList] = None
    toolResultList: Optional[ToolResultList] = None

    @model_validator(mode='after')
    def check_only_one_field(cls, values):
        fields = ['text', 'toolCallList', 'toolResultList']
        filled_fields = [field for field in fields if getattr(values, field) is not None]
        if len(filled_fields) != 1:
            raise ValueError("Only one of 'text', 'toolCallList', or 'toolResultList' must be provided.")
        return values

class FunctionTool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class Tool(BaseModel):
    function: FunctionTool

class CompletionRequest(BaseModel):
    modelUri: str
    completionOptions: CompletionOptions
    messages: List[Message]
    tools: Optional[List[Tool]] = None

# Response Models
class AlternativeStatus(str, Enum):
    UNSPECIFIED = "ALTERNATIVE_STATUS_UNSPECIFIED"
    PARTIAL = "ALTERNATIVE_STATUS_PARTIAL"
    TRUNCATED_FINAL = "ALTERNATIVE_STATUS_TRUNCATED_FINAL"
    FINAL = "ALTERNATIVE_STATUS_FINAL"
    CONTENT_FILTER = "ALTERNATIVE_STATUS_CONTENT_FILTER"
    TOOL_CALLS = "ALTERNATIVE_STATUS_TOOL_CALLS"

class Alternative(BaseModel):
    message: Message
    status: AlternativeStatus

class ContentUsage(BaseModel):
    inputTextTokens: str
    completionTokens: str
    totalTokens: str

class CompletionResponse(BaseModel):
    alternatives: List[Alternative]
    usage: ContentUsage
    modelVersion: str
    
class AdapterCompletionRequest(BaseModel):
    yaCompletionRequest: CompletionRequest
    folderId: str
    apiKey: str
    id: str

# Ошибка для не поддерживаемых параметров в формате OpenAI
class _UnsupportedParameterError(Exception):
    """Exception raised for unsupported parameters in OpenAI format."""
    def __init__(self, parameter: str):
        self.parameter = parameter
        self.message = f"Unsupported parameter: {parameter}"
        super().__init__(self.message)

# Embedding Models
class TextEmbeddingRequest(BaseModel):
    modelUri: str
    text: str

class TextEmbeddingResponse(BaseModel):
    embedding: List[float]
    numTokens: str
    modelVersion: str
    
# Tuned Classification
class TunedTextClassificationRequest(BaseModel):
    modelUri: str
    text: str

class TunedClassificationLabel(BaseModel):
    label: str
    confidence: str

class TunedTextClassificationResponse(BaseModel):
    predictions: List[TunedClassificationLabel]
    modelVersion: str
    
# Few Shot Classification
class ClassificationSample(BaseModel):
    text: str
    label: str

class FewShotTextClassificationRequest(BaseModel):
    modelUri: str
    taskDescription: str
    labels: List[str]
    text: str
    samples: Optional[List[ClassificationSample]] = None

class ClassificationLabel(BaseModel):
    label: str
    confidence: str

class FewShotTextClassificationResponse(BaseModel):
    predictions: List[ClassificationLabel]
    modelVersion: str

# Other
class GetModelsResponse(BaseModel):
    models: List[str]
    mappedModels: Dict[str, str]
    
YaCompletionRequestWithClassificatiors = Union[CompletionRequest, TunedTextClassificationRequest, FewShotTextClassificationRequest]