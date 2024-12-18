from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi import APIRouter, HTTPException, Request
import os, sys, time, json
from pydantic import ValidationError
from app.my_logger import logger
from dotenv import load_dotenv
import asyncio
import random
import string
import uuid
from functools import wraps
from typeguard import check_type

from openai import BadRequestError
from openai._exceptions import InternalServerError, APIStatusError

from app.yandex.models import (
    CompletionRequest as YaCompletionRequest,
    CompletionResponse as YaCompletionResponse,
    TextEmbeddingRequest as YaTextEmbeddingRequest,
    TextEmbeddingResponse as YaTextEmbeddingResponse,
    FewShotTextClassificationRequest as YaFewShotTextClassificationRequest,
    FewShotTextClassificationResponse as YaFewShotTextClassificationResponse,
    TunedTextClassificationRequest as YaTunedTextClassificationRequest,
    TunedTextClassificationResponse as YaTunedTextClassificationResponse,
    GetModelsResponse as YaGetModelsResponse,
    YaCompletionRequestWithClassificatiors,
    
    ToolResult as YaToolResult,
    ToolCall as YaToolCall,
    ToolCallList as YaToolCallList,
    Message as YaChatCompletionMessage,
    CompletionOptions as YaCompletionOptions
)

from openai.types.chat import (
    ChatCompletion as OpenAIChatCompletion,
    ChatCompletionChunk as OpenAIChatCompletionChunk,
    ChatCompletionMessage as OpenAIChatCompletionMessage,
)
from openai.types.embedding import Embedding as OpenAIEmbedding
from openai.types.embedding_model import EmbeddingModel as OpenAIEmbeddingModel
from openai.types.embedding_create_params import EmbeddingCreateParams as OpenAIEmbeddingCreateParams
from openai.types.create_embedding_response import CreateEmbeddingResponse as OpenAICreateEmbeddingResponse
from openai.types.chat.completion_create_params import CompletionCreateParams as OpenAICompletionCreateParams
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming as OpenAICompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming as OpenAICompletionCreateParamsStreaming
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam as OpenAIChatCompletionMessageParam
from openai import BadRequestError
from app.yandex.completions import (
    _adapt_openai_to_yc_completions,
    generate_yandexgpt_response
)
from app.yandex.embeddings import (
    _adapt_openai_to_yc_embeddings,
    generate_yandexgpt_embeddings_response_batch
)

load_dotenv()

GITHUB_SHA = os.getenv("GITHUB_SHA", "unknown_version")
GITHUB_REF = os.getenv("GITHUB_REF", "unknown_branch")

logger.configure(extra={
    "GITHUB_SHA": GITHUB_SHA,
    "GITHUB_REF": GITHUB_REF
})
logger.info("Index module initiated.")

index = APIRouter()

def handle_error(e, request_id):
    
    if isinstance(e, TypeError):
        logger.error(f"Ошибка типа: {str(e)}")
        return JSONResponse(status_code=422, content={
            "error": {
                "message": str(e),
                "type": "type_error",
                "param": None,
                "code": 422
            }
        })
    
    elif isinstance(e, ValidationError):
        logger.error(f"Ошибка валидации: {str(e)}")
        return JSONResponse(status_code=422, content={
            "error": {
                "message": str(e),
                "type": "validation_error",
                "param": None,
                "code": 422
            }
        })
    
    elif isinstance(e, HTTPException):
        logger.error(f"HTTP ошибка: {str(e)}")
        return JSONResponse(status_code=e.status_code, content={
            "error": {
                "message": str(e),
                "type": "http_error",
                "param": None,
                "code": e.status_code
            }
        })
    
    elif isinstance(e, APIStatusError):
        logger.error(f"OpenAI API ошибка: {str(e)}")
        return JSONResponse(status_code=e.status_code, content={
            "error": {
                "message": e.message,
                "type": type(e).__name__,
                "param": None,
                "code": e.status_code
            }
        })
        
    elif isinstance(e, InternalServerError):
        logger.error(f"InternalServerError ошибка: {str(e)}")
        return JSONResponse(status_code=e.status_code, content={
            "error": {
                "message": e.message,
                "type": type(e).__name__,
                "param": None,
                "code": e.status_code
            }
        })
    
    else:
        logger.critical(f"Неожиданная ошибка: {str(e)}")
        return JSONResponse(status_code=500, content={
            "error": {
                "message": f"An unexpected error occurred. {request_id=}",
                "type": "unexpected_error",
                "param": None,
                "code": 500
            }
        })

def handle_request(func):
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        request_id = uuid.uuid4()
        
        # Проверка наличия заголовка Authorization
        if "Authorization" not in request.headers:
            logger.error("Отсутствует заголовок Authorization")
            return JSONResponse(status_code=401, content={"error": "Authorization header is required"})
        
        with logger.contextualize(request_id=request_id):
            try:
                return await func(request, *args, **kwargs)
            except Exception as e:
                return handle_error(e, request_id)
            
    return wrapper

@index.post("/v1/chat/completions")
@handle_request
async def completion(request: Request):
    logger.debug(f"Получен запрос на генерацию в формате OpenAI. {request.method=}\n{request.url=}\n{request.headers=}\n{request.client.host=}\n{request.client.port=}")
    
    folder_id, yandex_api_key = _decode_openai_api_key(request)
    
    logger.info("Генерация текста в Foundational Models", extra={"folder_id": folder_id})
    
    oai_completion_request: OpenAICompletionCreateParams = await request.json()
    
    # TODO add validation
    #check_type(oai_completion_request, OpenAICompletionCreateParams)
    
    logger.debug(f"Data: {oai_completion_request}")
    
    yc_completion_request: YaCompletionRequestWithClassificatiors = await _adapt_openai_to_yc_completions(oai_completion_request, folder_id)
    
    return await generate_yandexgpt_response(yc_completion_request, folder_id, yandex_api_key)

@index.post("/v1/embeddings")
@handle_request
async def embeddings(request: Request):
    logger.debug(f"Получен запрос на эмбеддинг текста в формате OpenAI. {request.method=}\n{request.url=}\n{request.headers=}\n{request.client.host=}\n{request.client.port=}")
    
    folder_id, yandex_api_key = _decode_openai_api_key(request)
    
    logger.info("Генерация эмбеддинга в Foundational Models", extra={"folder_id": folder_id})
    
    body = await request.json()
    logger.debug(f"Body: {body}")
    
    oai_text_embedding_request: OpenAIEmbeddingCreateParams = body
    
    yc_text_embedding_requests: list[YaTextEmbeddingRequest] = await _adapt_openai_to_yc_embeddings(oai_text_embedding_request, folder_id)
    
    return await generate_yandexgpt_embeddings_response_batch(yc_text_embedding_requests, folder_id, yandex_api_key)

def _decode_openai_api_key(request):
    openai_api_key = request.headers.get("Authorization", "").split("Bearer ")[-1].strip()
    
    if not openai_api_key:
        logger.error("Пустой API ключ")
        raise HTTPException(status_code=401, detail="Invalid API key provided")
    
    logger.debug(f"OpenAI Api-key: {openai_api_key}")
    
    try:
        folder_id, yandex_api_key = openai_api_key.split("@")
        
        if not folder_id or not yandex_api_key:
            raise ValueError("Пустой folder_id или yandex_api_key")
            
    except ValueError as e:
        logger.error(f"Ошибка при разборе API ключа: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format. Expected format: 'folder_id@yandex_api_key'"
        )
    
    logger.debug(f"Folder ID: {folder_id}\nYandex Api-key: {yandex_api_key}")
    return folder_id, yandex_api_key


###########################################################
# Checkers
###########################################################

@index.get("/")
def root():
    return {"status": "Hello from Foundational Models Team! check .../docs for more info"}

@index.get("/health")
def health_check():
    return {"status": "healthy"}

@index.get("/readyz")
def readiness_probe():
    return {"status": "ready"}

@index.get("/livez")
def liveness_probe():
    return {"status": "alive"}

@index.get("/badge")
def get_badge():
    return RedirectResponse(f"https://img.shields.io/badge/status-healthy-green")

@index.get("/badge-sha")
def get_badge_sha():
    return RedirectResponse(f"https://img.shields.io/badge/sha-{GITHUB_SHA}-blue")

@index.get("/badge-ref")
def get_badge_ref():
    ref = GITHUB_REF.split('/')[-1]
    return RedirectResponse(f"https://img.shields.io/badge/ref-{ref}-blue")

@index.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def method_not_allowed(path):
    return HTTPException(status_code=405, detail="Method Not Allowed")