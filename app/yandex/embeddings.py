from fastapi import HTTPException
from fastapi.responses import StreamingResponse
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
    FunctionResult as YaFunctionResult,
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
from openai._exceptions import InternalServerError, APIStatusError

from app.my_logger import logger
from dotenv import load_dotenv
import os
import json
import httpx
import time
import random
import string
from tenacity import retry, stop_after_attempt, wait_exponential
from typeguard import check_type
from dataclasses import dataclass
from typing import Optional
import asyncio
from asyncio import Queue

load_dotenv()

YC_EMBEDDINGS_MODEL_MAP = os.getenv("YC_EMBEDDINGS_MODEL_MAP", "gpt-4o:yandexgpt/latest,gpt-4o-mini:yandexgpt-lite/latest,gpt-3.5:yandexgpt/latest,gpt-3.5-turbo:yandexgpt/latest,gpt-5:yandexgpt/latest")
YC_LOG_POLICY = os.getenv("YC_FOMO_LOG_POLICY", "True").lower() == "true"
YC_SERVICE_URL = os.getenv("YC_SERVICE_URL", "https://llm.api.cloud.yandex.net")
YC_EMBEDDINGS_RETRIES = os.getenv("YC_EMBEDDINGS_RETRIES", "True").lower() == "true"
YC_EMBEDDINGS_RATE_LIMIT = int(os.getenv("YC_EMBEDDINGS_RATE_LIMIT", "1"))
YC_EMBEDDINGS_TIME_WINDOW = int(os.getenv("YC_EMBEDDINGS_TIME_WINDOW", "1"))
YC_EMBEDDINGS_MAX_RETRIES = int(os.getenv("YC_EMBEDDINGS_MAX_RETRIES", "1"))
YC_EMBEDDINGS_BACKOFF_FACTOR = int(os.getenv("YC_EMBEDDINGS_BACKOFF_FACTOR", "1"))

try:
    embeddings_model_map = {k: v for k, v in [item.split(":") for item in YC_EMBEDDINGS_MODEL_MAP.split(",")]}
except Exception as e:
    logger.error(f"Error parsing YC_EMBEDDINGS_MODEL_MAP: {e}")
    raise e

UNSUPPORTED_PARAMETERS = {
    "dimensions",
    "encoding_format",
    "user"
}

async def send_request(url: str, headers: dict, body: str, timeout: int = 60):
    if YC_EMBEDDINGS_RETRIES:
    #if False:
        return await send_request_with_retry(url, headers, body, timeout)
    else:
        return await send_request_without_retry(url, headers, body, timeout)

async def send_request_without_retry(url, headers, body, timeout):
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, content=body, timeout=timeout)
        return response
    
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def send_request_with_retry(url, headers, body, timeout):
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, content=body, timeout=timeout)
        return response

def _prepare_request_headers(api_key: str, folder_id: str) -> dict:
    """Подготавливает заголовки запроса."""
    auth_value = f"Bearer {api_key}" if api_key.startswith('t1') else f"Api-Key {api_key}"
    return {
        "Content-Type": "application/json",
        "Authorization": auth_value,
        'x-folder-id': folder_id,
        'x-data-logging-enabled': str(YC_LOG_POLICY)
    }

async def _validate_response(response: httpx.Response):
    """Проверяет корректность ответа."""
    if response.status_code != 200:
        error_msg = f"Ошибка API: {response.text}"
        logger.error(error_msg)
        raise HTTPException(status_code=response.status_code, detail=error_msg)

def _construct_model_name(
    yc_text_embedding_request: YaTextEmbeddingRequest,  
    yandex_response: YaTextEmbeddingResponse
) -> str:
    """Формирует имя модели в формате OpenAI."""
    model_base = yc_text_embedding_request.modelUri.split('/')[-2]
    model_version = yandex_response.modelVersion
    return f"{model_base}-by-{model_version}"

def _log_warning_on_unsupported_parameters(parameters: OpenAIEmbeddingCreateParams, unsupported_parameters: set[str]):
    input_parameters = set(parameters.keys())
    unsupported_parameters_in_input = input_parameters.intersection(unsupported_parameters)
    
    if unsupported_parameters_in_input:
        logger.warning(f"Unsupported parameters in input: {unsupported_parameters_in_input}")

async def _adapt_openai_to_yc_embeddings(oai_text_embedding_request: OpenAIEmbeddingCreateParams, folder_id: str) -> list[YaTextEmbeddingRequest]:
    logger.debug(f"Transforming OpenAI embeddings request to Yandex GPT request_S. OaiCompletionRequest: {oai_text_embedding_request}, folder_id: {folder_id}")
    
    logger.info("Модель для генерации эмбеддингов в Foundational Models", extra={
        "model": str(oai_text_embedding_request.get("model")), 
        "folder_id": folder_id
    })
    
    model_uri: str = _get_embeddings_model_uri(oai_text_embedding_request.get("model"), folder_id)
    
    _log_warning_on_unsupported_parameters(oai_text_embedding_request, UNSUPPORTED_PARAMETERS)
    
    input_texts: list[str] = oai_text_embedding_request.get("input")
    
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    
    _warning_on_enourmos_amount_of_texts(input_texts)
    
    yandex_requests = [YaTextEmbeddingRequest(
        modelUri=model_uri,
        text=text
    ) for text in input_texts]
    
    logger.debug(f"Transformed Yandex request: {str(yandex_requests)[:100]}")
    return yandex_requests

def _warning_on_enourmos_amount_of_texts(input_texts: list[str]):
    if len(input_texts) > 1000:
        logger.warning(f"Large amount of texts: {len(input_texts)}")

async def generate_yandexgpt_embeddings_response_batch(
    yc_text_embedding_requests: list[YaTextEmbeddingRequest],
    folder_id: str,
    yandex_api_key: str
) -> OpenAICreateEmbeddingResponse:
    logger.debug(
        f"Sending Yandex embeddings request to Yandex GPT. "
        f"Yandex embeddings request: {str(yc_text_embedding_requests)[:100]}, "
        f"folder_id: {folder_id}, Api-key: {yandex_api_key}"
    )
    
    if not yc_text_embedding_requests:
        raise HTTPException(status_code=400, detail="Empty request list")
        
    url = f"{YC_SERVICE_URL}/foundationModels/v1/textEmbedding"
    headers = _prepare_request_headers(yandex_api_key, folder_id)
    
    all_embeddings = []
    all_token_counts = 0
    last_response = None
    
    RATE_LIMIT = YC_EMBEDDINGS_RATE_LIMIT
    TIME_WINDOW = YC_EMBEDDINGS_TIME_WINDOW
    MAX_RETRIES = YC_EMBEDDINGS_MAX_RETRIES
    BACKOFF_FACTOR = YC_EMBEDDINGS_BACKOFF_FACTOR

    queue = Queue()
    retry_queue = Queue()
    for request in yc_text_embedding_requests:
        queue.put_nowait((request, 0))  # (request, retry_count)

    async def worker():
        nonlocal all_embeddings, all_token_counts, last_response
        while not queue.empty() or not retry_queue.empty():
            try:
                # Сначала проверяем очередь повторных попыток
                if not retry_queue.empty():
                    request, retry_count = await retry_queue.get()
                else:
                    request, retry_count = await queue.get()

                try:
                    body = request.model_dump_json()
                    response = await send_request(url, headers, body, timeout=119)
                    await _validate_response(response)
                    
                    result = response.json()
                    logger.debug(f"Yandex response: {str(result)[:100]}")
                    
                    yandex_response = YaTextEmbeddingResponse(**result)
                    last_response = yandex_response
                    all_token_counts += int(yandex_response.numTokens)
                    
                    embedding = OpenAIEmbedding(
                        embedding=yandex_response.embedding,
                        index=len(all_embeddings),
                        object="embedding"
                    )
                    all_embeddings.append(embedding)

                except HTTPException as e:
                    if e.status_code == 429 and retry_count < MAX_RETRIES:
                        # При ошибке rate limit добавляем запрос обратно с увеличенной задержкой
                        await asyncio.sleep(TIME_WINDOW * BACKOFF_FACTOR ** retry_count)
                        await retry_queue.put((request, retry_count + 1))
                        logger.warning(f"Rate limit hit, retrying request (attempt {retry_count + 1})")
                    else:
                        logger.error(f"Error during processing: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Error during processing: {e}")
                    raise

            finally:
                if not retry_queue.empty():
                    retry_queue.task_done()
                else:
                    queue.task_done()

    try:
        async with asyncio.timeout(120):  # 2 минуты максимум
            workers = []
            for _ in range(RATE_LIMIT):
                worker_coroutine = worker()
                workers.append(asyncio.create_task(worker_coroutine))
            
            # Ждем завершения обеих очередей
            await queue.join()
            await retry_queue.join()
            
            # Отменяем оставшиеся задачи
            for w in workers:
                w.cancel()
            
            # Ждем отмены всех задач
            await asyncio.gather(*workers, return_exceptions=True)
            
    except TimeoutError:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise HTTPException(status_code=504, detail="Request timeout")
    
    if not all_embeddings:
        raise HTTPException(status_code=500, detail="No embeddings were generated")
    
    final = OpenAICreateEmbeddingResponse(
        data=all_embeddings,
        model=_construct_model_name(yc_text_embedding_requests[0], last_response),
        object="list",
        usage={
            "prompt_tokens": all_token_counts,
            "total_tokens": all_token_counts
        }
    )
    
    await _log_success_on_embeddings(final, folder_id, yc_text_embedding_requests, last_response)
    return final

async def _log_success_on_embeddings(final: OpenAICreateEmbeddingResponse, folder_id: str, yc_text_embedding_requests: list[YaTextEmbeddingRequest], last_response: YaTextEmbeddingResponse):
    yc_text_embedding_request = yc_text_embedding_requests[0]
    
    logger.success("Генерация текста в Foundational Models завершена", 
        extra={
            "folder_id": folder_id, 
            "modelUri": yc_text_embedding_request.modelUri,
            "modelPrefix": yc_text_embedding_request.modelUri.split(":")[0],
            "model": yc_text_embedding_request.modelUri.split("/")[-2:],
            "total_texts": len(yc_text_embedding_requests),
            "total_tokens": final.usage.total_tokens,
            "model_version": last_response.modelVersion,
            "openai_model": final.model,
        })

def _get_embeddings_model_uri(model: str, folder_id: str) -> str:
    """
    1. map model to yc model
    2. check clf mode, raise clf
    3. construct yc model uri
    """
    logger.debug(f"Model: {model}, folder_id: {folder_id}")
    
    # map model to yc model
    if model in embeddings_model_map:
        model = embeddings_model_map[model]
    
    if model.startswith(("emb://")):
        model_uri = model
    else:
        model_uri = f"emb://{folder_id}/{model}"
    
    logger.debug(f"Model URI: {model_uri}")
    return model_uri