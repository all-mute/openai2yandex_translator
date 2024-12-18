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

load_dotenv()

YC_COMPLETIONS_MODEL_MAP = os.getenv("YC_COMPLETIONS_MODEL_MAP", "gpt-4o:yandexgpt/latest,gpt-4o-mini:yandexgpt-lite/latest,gpt-3.5:yandexgpt/latest,gpt-3.5-turbo:yandexgpt/latest,gpt-5:yandexgpt/latest")
YC_LOG_POLICY = os.getenv("YC_FOMO_LOG_POLICY", "True").lower() == "true"
YC_SERVICE_URL = os.getenv("YC_SERVICE_URL", "https://llm.api.cloud.yandex.net")
YC_COMPLETION_RETRIES = os.getenv("YC_COMPLETION_RETRIES", "True").lower() == "true"

try:
    completions_model_map = {k: v for k, v in [item.split(":") for item in YC_COMPLETIONS_MODEL_MAP.split(",")]}
except Exception as e:
    logger.error(f"Error parsing YC_COMPLETIONS_MODEL_MAP: {e}")
    raise e

UNSUPPORTED_PARAMETERS = {
    # "messages",
    # "model",
    # "stream",
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    # "max_completion_tokens",
    # "max_tokens",
    "metadata",
    "modalities",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "response_format",
    "seed",
    "service_tier",
    # "stop",
    "store",
    "stream_options",
    # "temperature",
    "tool_choice",
    # "tools",
    "top_logprobs",
    "top_p",
    "user"
}

async def send_request(url: str, headers: dict, body: str, timeout: int = 60):
    if YC_COMPLETION_RETRIES:
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

async def _adapt_openai_to_yc_completions(
    oai_completion_request: OpenAICompletionCreateParams,
    folder_id: str
) -> YaCompletionRequestWithClassificatiors:
    
    logger.debug(f"Transforming OpenAI completion request to Yandex GPT request. OaiCompletionRequest: {oai_completion_request}, folder_id: {folder_id}")
    
    logger.info("Модель для генерации текста в Foundational Models", extra={
        "model": str(oai_completion_request.get("model")), 
        "using_tools": str(bool(oai_completion_request.get("tools"))),
        "folder_id": folder_id
    })
    
    model_uri = _get_completions_model_uri(oai_completion_request.get("model"), folder_id)
    
    _log_warning_on_unsupported_parameters(oai_completion_request, UNSUPPORTED_PARAMETERS)
    
    if model_uri.startswith("cls://"):
        pass
    else: 
        yandex_parameters = YaCompletionOptions(
            stream=oai_completion_request.get("stream"),
            temperature=oai_completion_request.get("temperature"),
            maxTokens=_get_max_tokens(oai_completion_request)
        )
        yandex_messages = _adapt_messages(oai_completion_request.get("messages"))
        yandex_tools = oai_completion_request.get("tools")
        
        yandex_request = YaCompletionRequest(
            modelUri=model_uri,
            messages=yandex_messages,
            completionOptions=yandex_parameters,
            tools=yandex_tools
        )
        
        logger.debug(f"Transformed Yandex request: {yandex_request}")
        return yandex_request
    
    
async def generate_yandexgpt_response(
    yc_completion_request: YaCompletionRequestWithClassificatiors,
    folder_id: str,
    yandex_api_key: str
) -> YaCompletionResponse | YaTunedTextClassificationResponse | YaFewShotTextClassificationResponse:
    logger.debug(f"Sending Yandex completion request to Yandex GPT. Yandex completion request: {yc_completion_request}, folder_id: {folder_id}, Api-key: {yandex_api_key}")
    
    if isinstance(yc_completion_request, YaCompletionRequest):
        logger.debug("Choosing completion response")
        return await generate_yandexgpt_completion_response(yc_completion_request, folder_id, yandex_api_key)
    elif isinstance(yc_completion_request, YaTunedTextClassificationRequest):
        logger.debug("Choosing tuned text classification response")
        pass
        # return tuned text classification response
    elif isinstance(yc_completion_request, YaFewShotTextClassificationRequest):
        logger.debug("Choosing few shot text classification response")
        pass
        # return few shot text classification response


async def generate_yandexgpt_completion_response(
    yc_completion_request: YaCompletionRequest,
    folder_id: str,
    yandex_api_key: str
) -> YaCompletionResponse:
    if yc_completion_request.completionOptions.stream:
        return StreamingResponse(generate_yandexgpt_completion_response_streaming_with_tools(yc_completion_request, folder_id, yandex_api_key), media_type="text/event-stream")
    else:
        return await generate_yandexgpt_completion_response_non_streaming(yc_completion_request, folder_id, yandex_api_key)
        
async def generate_yandexgpt_completion_response_non_streaming(
    yc_completion_request: YaCompletionRequest,
    folder_id: str,
    yandex_api_key: str
) -> OpenAIChatCompletion:
    logger.debug("Generating non streaming completion response")
    
    url = f"{YC_SERVICE_URL}/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {yandex_api_key}" if yandex_api_key.startswith('t1') else f"Api-Key {yandex_api_key}",
        'x-folder-id': folder_id,
        'x-data-logging-enabled': str(YC_LOG_POLICY)
    }
    body = yc_completion_request.model_dump_json()
    
    logger.debug(f"Отправка запроса на {url} с заголовками: {headers} и данными: {body}")
    response: httpx.Response = await send_request(url, headers, body, 60)
    
    if response.status_code == 200:
        logger.debug(f"Получен ответ от Yandex GPT: {response.text}, {response.status_code}, {response.headers}")
        
        result = response.json().get('result')
        yandex_completion_response = YaCompletionResponse(**result)
        
        final_result: OpenAIChatCompletion = _transform_to_openai_response_format(yandex_completion_response, yc_completion_request, response.headers)
        
        logger.debug(f"Final result: {final_result}")
        _log_success_on_completion(final_result, yandex_completion_response, folder_id, yc_completion_request)
        
        return final_result
    else:
        logger.error(f"Error generating completion response: {response.status_code}, {response.text}, {response.headers}")
        logger.info(f"Error generating completion response", extra={
            "folder_id": folder_id, 
            "modelUri": yc_completion_request.modelUri,
            "model": yc_completion_request.modelUri.split("/")[-2:],
            "error_code": response.status_code,
            "error_message": response.text
        })
        
        # TODO: map errors into OpenAI format with more details/codes/errors
        if str(response.status_code).startswith("4"):
            raise APIStatusError(message=response.text, response=response, body=response.text)
        elif str(response.status_code).startswith("5"):
            raise InternalServerError(message=response.text, response=response, body=response.text)
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    
async def generate_yandexgpt_completion_response_streaming_with_tools(
    yc_completion_request: YaCompletionRequest,
    folder_id: str, 
    yandex_api_key: str
):
    headers = _prepare_request_headers(yandex_api_key, folder_id)
    request_body = yc_completion_request.model_dump_json()
    
    logger.debug(f"Отправка стрим-запроса: URL={YC_SERVICE_URL}, headers={headers}")
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream('POST', 
                                  f"{YC_SERVICE_URL}/foundationModels/v1/completion",
                                  headers=headers,
                                  content=request_body,
                                  timeout=30.0) as response:
                
                await _validate_response(response)
                
                accumulated_text = ""
                last_chunk = None
                async for chunk in _process_response_stream(response, yc_completion_request):
                    yield chunk
                    if isinstance(chunk, str) and "data: [DONE]" in chunk:
                        if last_chunk and isinstance(last_chunk, YaCompletionResponse):
                            _log_success_on_streaming_completion(
                                last_chunk, 
                                folder_id, 
                                yc_completion_request
                            )
                    else:
                        last_chunk = chunk
                    
    except httpx.TimeoutException:
        logger.error("Таймаут при получении ответа от YandexGPT")
        raise HTTPException(status_code=504, detail="Gateway Timeout")
    except Exception as e:
        logger.error(f"Ошибка при стриминге ответа: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _log_success_on_streaming_completion(
    yandex_completion_response: YaCompletionResponse, 
    folder_id: str, 
    yc_completion_request: YaCompletionRequest
):
    """Логирует успешное завершение стримингового запроса."""
    logger.success("Стриминговая генерация текста в Foundational Models завершена", 
                    extra={
                        "folder_id": folder_id, 
                        "modelUri": yc_completion_request.modelUri,
                        "modelPrefix": yc_completion_request.modelUri.split(":")[0],
                        "model": yc_completion_request.modelUri.split("/")[-2:],
                        "streaming": yc_completion_request.completionOptions.stream,
                        "input_text_tokens": yandex_completion_response.usage.inputTextTokens,
                        "completion_tokens": yandex_completion_response.usage.completionTokens,
                        "total_tokens": yandex_completion_response.usage.totalTokens,
                        "model_version": yandex_completion_response.modelVersion,
                        "is_toolResult": str(bool(yandex_completion_response.alternatives[0].message.toolResultList)),
                        "yandex_status": str(yandex_completion_response.alternatives[0].status)
                    })

async def _process_response_stream(response: httpx.Response, yc_completion_request: YaCompletionRequest):
    """Обрабатывает поток ответов от API."""
    accumulated_text = ""
    
    async for line in response.aiter_lines():
        if not line:
            continue
            
        try:
            chunk = _parse_response_chunk(line, accumulated_text)
            if chunk.is_tool_call:
                yield _format_tool_call_response(chunk, yc_completion_request)
                yield "data: [DONE]\n\n"
                return
                
            accumulated_text = chunk.accumulated_text
            yield _format_text_chunk_response(chunk)
            
            if chunk.is_complete:
                yield "data: [DONE]\n\n"
                return
                
        except json.JSONDecodeError:
            logger.error(f"Ошибка парсинга JSON из строки: {line}")
            continue

@dataclass
class ResponseChunk:
    content: str
    is_complete: bool
    is_tool_call: bool
    accumulated_text: str
    response_obj: YaCompletionResponse

def _parse_response_chunk(line: str, accumulated_text: str) -> ResponseChunk:
    """Парсит чанк ответа."""
    json_data = json.loads(line)
    response_obj = YaCompletionResponse(**json_data['result'])
    
    is_tool_call = (response_obj.alternatives[0].status == 
                    "ALTERNATIVE_STATUS_TOOL_CALLS")
    
    if is_tool_call:
        return ResponseChunk(
            content="",
            is_complete=True,
            is_tool_call=True,
            accumulated_text=accumulated_text,
            response_obj=response_obj
        )
        
    new_text = response_obj.alternatives[0].message.text
    content = new_text[len(accumulated_text):]
    
    return ResponseChunk(
        content=content,
        is_complete=response_obj.alternatives[0].status == "ALTERNATIVE_STATUS_COMPLETE",
        is_tool_call=False,
        accumulated_text=new_text,
        response_obj=response_obj
    )

def _format_text_chunk_response(chunk: ResponseChunk) -> str:
    """Форматирует текстовый чанк в формат SSE."""
    response_data = {
        "id": _generate_completion_id(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "yandexgpt-latest",
        "system_fingerprint": _generate_fingerprint(),
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": chunk.content
            },
            "logprobs": None,
            "finish_reason": "stop" if chunk.is_complete else None
        }]
    }
    
    return f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"

def _generate_completion_id():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def _generate_fingerprint():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def _format_tool_call_response(chunk: ResponseChunk, yc_completion_request: YaCompletionRequest) -> str:
    """Форматирует ответ с вызовом инструмента в формат SSE."""
    response_data = _transform_to_openai_response_format(
        chunk.response_obj,
        yc_completion_request,
        {}  # headers не используются для tool calls
    )
    return f"data: {json.dumps(response_data.model_dump(), ensure_ascii=False)}\n\n"

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

def _log_success_on_completion(final_result: OpenAIChatCompletion, yandex_completion_response: YaCompletionResponse, folder_id: str, yc_completion_request: YaCompletionRequest):
    logger.success("Генерация текста в Foundational Models завершена", 
                    extra={
                        "folder_id": folder_id, 
                        "modelUri": yc_completion_request.modelUri,
                        "modelPrefix": yc_completion_request.modelUri.split(":")[0],
                        "model": yc_completion_request.modelUri.split("/")[-2:],
                        "streaming": yc_completion_request.completionOptions.stream,
                        "input_text_tokens": yandex_completion_response.usage.inputTextTokens,
                        "completion_tokens": yandex_completion_response.usage.completionTokens,
                        "total_tokens": yandex_completion_response.usage.totalTokens,
                        "model_version": yandex_completion_response.modelVersion,
                        "is_toolResult": str(bool(yandex_completion_response.alternatives[0].message.toolResultList)),
                        "yandex_status": str(yandex_completion_response.alternatives[0].status),
                        "openai_status": final_result.choices[0].finish_reason,
                        "openai_id": final_result.id,
                        "openai_created": final_result.created,
                        "openai_model": final_result.model,
                        "openai_system_fingerprint": final_result.system_fingerprint
                    })
    
def _transform_to_openai_response_format(
    yandex_response: YaCompletionResponse,
    yc_completion_request: YaCompletionRequest,
    headers: dict
) -> OpenAIChatCompletion:
    """
    Преобразует ответ Yandex GPT в формат ответа OpenAI.

    Args:
        yandex_response (YaCompletionResponse): Ответ от Yandex GPT.
        yc_completion_request (YaCompletionRequest): Исходный запрос к Yandex GPT.
        headers (dict): Заголовки запроса.

    Returns:
        OpenAIChatCompletion: Ответ в формате OpenAI.
    """
    try:
        request_id = headers.get('x-request-id', _generate_fallback_id())
        is_tool_call = _check_if_tool_call(yandex_response)
        
        if is_tool_call:
            tool_calls = _extract_tool_calls(yandex_response)
            finish_reason = "tool_calls"
            content = None
        else:
            tool_calls = None
            finish_reason = "stop"
            content = _extract_content(yandex_response)
        
        openai_response = OpenAIChatCompletion(
            id=_generate_completion_id(),
            object="chat.completion",
            created=int(time.time()),
            model=_construct_model_name(yc_completion_request, yandex_response),
            system_fingerprint=request_id,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls
                    },
                    "logprobs": None,
                    "finish_reason": finish_reason
                }
            ],
            usage={
                "prompt_tokens": yandex_response.usage.inputTextTokens,
                "completion_tokens": yandex_response.usage.completionTokens,
                "total_tokens": yandex_response.usage.totalTokens
            }
        )

        logger.debug(f"Формирование ответа в формате OpenAI завершено, ответ: {openai_response}")
        return openai_response

    except Exception as e:
        logger.error(f"Ошибка при преобразовании ответа: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при обработке ответа от Yandex GPT")


def _generate_completion_id(length: int = 8) -> str:
    """Генерирует уникальный идентификатор для завершения."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def _generate_fallback_id() -> str:
    """Генерирует запасной идентификатор, если отсутствует в заголовках."""
    return _generate_completion_id()


def _check_if_tool_call(yandex_response: YaCompletionResponse) -> bool:
    """Определяет, содержит ли ответ вызовы инструментов."""
    return yandex_response.alternatives[0].status == "ALTERNATIVE_STATUS_TOOL_CALLS"


def _extract_tool_calls(yandex_response: YaCompletionResponse) -> Optional[list]:
    """Извлекает информацию о вызовах инструментов из ответа."""
    try:
        return [
            {
                "id": f"call_{_generate_completion_id()}",
                "type": "function",
                "function": {
                    "name": tool_call.functionCall.name,
                    "arguments": json.dumps(tool_call.functionCall.arguments)
                }
            }
            for tool_call in yandex_response.alternatives[0].message.toolCallList.toolCalls
        ]
    except AttributeError as e:
        logger.error(f"Ошибка при извлечении вызовов инструментов: {e}")
        return None


def _extract_content(yandex_response: YaCompletionResponse) -> Optional[str]:
    """Извлекает текстовое содержимое из ответа."""
    try:
        return yandex_response.alternatives[0].message.text
    except AttributeError as e:
        logger.error(f"Ошибка при извлечении содержимого: {e}")
        return None


def _construct_model_name(
    yc_completion_request: YaCompletionRequest,
    yandex_response: YaCompletionResponse
) -> str:
    """Формирует имя модели в формате OpenAI."""
    model_base = yc_completion_request.modelUri.split('/')[-2]
    model_version = yandex_response.modelVersion
    return f"{model_base}-by-{model_version}"

def _get_completions_model_uri(model: str, folder_id: str) -> str:
    """
    1. map model to yc model
    2. check clf mode, raise clf
    3. construct yc model uri
    """
    logger.debug(f"Model: {model}, folder_id: {folder_id}")
    
    # map model to yc model
    if model in completions_model_map:
        model = completions_model_map[model]
    
    if model.startswith(("gpt://", "ds://")):
        model_uri = model
    elif model.startswith("cls://"):
        raise HTTPException(status_code=400, detail="Classifier mode is not supported yet")
        model_uri = model
    else:
        model_uri = f"gpt://{folder_id}/{model}"
    
    logger.debug(f"Model URI: {model_uri}")
    return model_uri

def _adapt_messages(messages: list[OpenAIChatCompletionMessageParam]) -> list[YaChatCompletionMessage]:
    logger.debug(f"Messages: {messages}")
    
    messages_transformed = []
    called_functions = {}
    
    i = 0
    while i < len(messages):
        logger.debug(f"Processing message {i+1} of {len(messages)}")
        message = messages[i]
        
        try:
            
            if message.get('role') == 'function' or (message.get('role') == 'assistant' and message.get('function_call')):
                _raise_deprecated_function_call_error()
            
            if message.get('role') == 'tool':
                toolResults, i = _collect_tool_results(messages, i, called_functions)
                messages_transformed.append(YaChatCompletionMessage(
                    role="assistant",
                    toolResultList={
                        "toolResults": toolResults
                    }
                ))
            elif message.get('role') == 'assistant' and message.get('tool_calls'):
                toolCalls = _process_tool_calls(message, called_functions)
                messages_transformed.append(YaChatCompletionMessage(
                    role="assistant",
                    toolCallList={
                        "toolCalls": toolCalls
                    }
                ))
            else:
                content = _get_content_as_string(message.get('content'))
                yc_message = YaChatCompletionMessage(
                    role=message.get('role'),
                    text=content
                )
                messages_transformed.append(yc_message)
        
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {message}, ошибка: {e}")
            raise e
        
        i += 1
    
    logger.debug(f"Преобразование сообщений в формат Yandex GPT завершено, результат: {messages_transformed}")
    return messages_transformed

def _collect_tool_results(messages: list[OpenAIChatCompletionMessageParam], start_index: int, called_functions: dict):
    logger.debug(f"Collecting tool results from messages: {messages}, start_index: {start_index}, called_functions: {called_functions}")
    toolResults = []
    i = start_index
    while i < len(messages) and messages[i].get('role') == 'tool':
        message = messages[i]
        name = called_functions.get(message.get('tool_call_id'))
        toolResults.append(YaToolResult(
            functionResult=YaFunctionResult(
                name=name,
                content=message.get('content')
            )
        ))
        i += 1
    return toolResults, i - 1 

def _process_tool_calls(message: OpenAIChatCompletionMessageParam, called_functions: dict):
    logger.debug(f"Processing tool calls from message: {message}, called_functions: {called_functions}")
    toolCalls = []
    for tool_call in message.get('tool_calls'):
        try:
            name = tool_call.get('function').get('name')
            arguments = json.loads(tool_call.get('function').get('arguments'))
            toolCalls.append(YaToolCall(
                functionCall={
                    "name": name,
                    "arguments": arguments
                }
            ))
            called_functions[tool_call.get('id')] = name
        except Exception as e:
            logger.error(f"Ошибка при извлечении name и arguments из tool_call: {tool_call}, ошибка: {e}")
    return toolCalls

def _get_content_as_string(content):
    if content and not isinstance(content, str):
        return str(content)
    return content

def _raise_deprecated_function_call_error():
    raise HTTPException(status_code=400, detail="Function calling is deprecated and not supported by OpenAI API to Yandex GPT Adapter. Use tool calling instead.")

def _log_warning_on_unsupported_parameters(parameters: OpenAICompletionCreateParams, unsupported_parameters: set[str]):
    input_parameters = set(parameters.keys())
    unsupported_parameters_in_input = input_parameters.intersection(unsupported_parameters)
    
    if unsupported_parameters_in_input:
        logger.warning(f"Unsupported parameters in input: {unsupported_parameters_in_input}")

def _get_max_tokens(completion_request: OpenAICompletionCreateParams) -> str | None:
    """
    Извлекает и обрабатывает параметр максимального количества токенов из запроса.
    
    Args:
        completion_request: Параметры запроса в формате OpenAI
        
    Returns:
        str | None: Строковое представление max_tokens или None, если не задано
    """
    max_completion_tokens = completion_request.get("max_completion_tokens")
    max_tokens = completion_request.get("max_tokens")
    
    # Приоритет отдается max_completion_tokens, если он задан
    final_max_tokens = max_completion_tokens or max_tokens
    
    return str(final_max_tokens) if final_max_tokens is not None else None