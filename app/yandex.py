import string
import httpx
import time, json
from fastapi import HTTPException
from app.my_logger import logger
from app.models import CompletionResponse, TextEmbeddingResponse, CompletionRequest, TextEmbeddingRequest
from app.metrics import increment_yandex_metric_counter
import random
from collections import defaultdict, deque

async def generate_yandexgpt_stream_response(messages, tools, model: str, temperature: float, max_tokens: int, yandex_api_key: str, folder_id: str):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {yandex_api_key}" if yandex_api_key.startswith('t1') else f"Api-Key {yandex_api_key}",
        'x-folder-id': folder_id
    }
    
    logger.debug("Начинаем преобразование сообщений в формат Yandex GPT.")
            
    messages_transformed, called_functions = _adapt_messages_for_yandexgpt(messages, tools)

    # Определение URI модели
    model_uri = _get_completions_model_uri(model, folder_id)
    logger.debug(f"Определён URI модели: {model_uri}")
    
    # Формирование запроса
    payload = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": True,
            "temperature": temperature,
            "maxTokens": max_tokens
        },
        "messages": messages_transformed,
        "tools": tools
    }
    
    logger.info("Формирование запроса завершено, отправка запроса в Yandex GPT.")
    
    generated_len = 0
    finished = False
    
    logger.info("Отправка запроса в Yandex GPT...")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=30)  # Увеличение таймаута
        logger.debug(f"Ответ от Yandex GPT получен с кодом статуса: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Ошибка при отправке запроса: {response.text}, статус код: {response.status_code}")
            return
        
        async for line in response.aiter_lines():
            if line:
                logger.debug(f"Полученная строка: {line}")
                json_line = json.loads(line)  # Преобразование ��троки в JSON
                
                if json_line['result']['alternatives'][0]['status'] == "ALTERNATIVE_STATUS_TOOL_CALLS":
                    logger.debug("Получен инструмент!")
                    response_obj = CompletionResponse(**json_line['result'])
                    logger.debug(f"Преобразованный объект: {response_obj}")
                    
                    response_data = _adapt_message_for_openai_stream_tool_calls(response_obj, model)
                    
                    yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                    
                    finished = True
                    break
                    
                if 'result' in json_line:
                    response_obj = CompletionResponse(**json_line['result'])
                    logger.debug(f"Преобразованный объект: {response_obj}")

                    content = response_obj.alternatives[0].message.text[generated_len:]
                    generated_len = len(response_obj.alternatives[0].message.text)

                    response_data = {
                        "id": "chatcmpl-42",  # Здесь можно использовать уникальный ID
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "gpt-4o-mini-2024-07-18",
                        "system_fingerprint": "fp_e2bde53e6",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",  # Добавляем роль
                                    "content": content
                                },
                                "logprobs": None,
                                "finish_reason": None if response_obj.alternatives[0].status == "ALTERNATIVE_STATUS_PARTIAL" else "stop"
                            }
                        ]
                    }

                    logger.debug(f"Отправка данных: {response_data}")
                    yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"

                if response_obj.alternatives[0].status == "ALTERNATIVE_STATUS_COMPLETE":
                    finished = True
                    logger.info("Получен полный ответ от Yandex GPT.")
                    await increment_yandex_metric_counter("yandexgpt_completions_requests", labels={"model": model})
                    
                    break

    if finished:
        logger.info("Завершение передачи данных.")
        yield "data: [DONE]\n\n"

                    

async def generate_yandexgpt_response(messages, tools, model: str, temperature: float, max_tokens: int, yandex_api_key: str, folder_id: str) -> tuple[CompletionResponse, None] | tuple[None, dict]:
    """
    Генерация ответа от Yandex GPT.

    :param messages: Список сообщений для генерации ответа.
    :param model: Модель, используемая для генерации.
    :param temperature: Температура для генерации.
    :param max_tokens: Максимальное количество токенов в ответе.
    :param yandex_api_key: API ключ для доступа к Yandex.
    :param folder_id: ID папки в Yandex.
    :return: Ответ от Yandex GPT или ошибка.
    """
    logger.debug("Начало генерации ответа от Yandex GPT.")
    
    async def _send_request_safety(payload: dict, api_key: str) -> tuple[CompletionResponse, None] | tuple[None, dict]:
        """
        Отправка запроса в Yandex GPT с обработкой ошибок.

        :param payload: Данные з��проса.
        :param api_key: API ключ для авторизации.
        :return: Ответ от Yandex GPT или ошибка.
        """
        try:
            url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key.startswith('t1') else f"Api-Key {api_key}",
                'x-folder-id': folder_id
            }

            logger.debug(f"Отправка запроса на {url} с заголовками: {headers} и данными: {payload}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                logger.debug(f"Ответ от Yandex GPT: {response.json()}")
                
                response_obj = CompletionResponse(**response.json()['result'])
                
                logger.info("Запрос успешно выполнен, получен ответ от Yandex GPT.")
                await increment_yandex_metric_counter("yandexgpt_completions_requests", labels={"model": model})
                
                return response_obj, None
            
            else:
                logger.error(f"Ошибка при выполнении запроса: {response.text}, статус код: {response.status_code}")
                
                return None, {
                    "error": {
                        "message": f"Ошибка: {response.text}",
                        "type": "api_error",
                        "param": None,
                        "code": response.status_code
                    }
                }
        except httpx.ReadTimeout:
            logger.error("Таймаут чтения: запрос не завершился вовремя.")
            return None, {"error": "Таймаут чтения: запрос не завершился вовремя."}
            
    messages_transformed, called_functions = _adapt_messages_for_yandexgpt(messages, tools)

    # Определение URI модели
    model_uri = _get_completions_model_uri(model, folder_id)
    logger.debug(f"Определён URI модели: {model_uri}")
    
    # Формирование запроса
    payload = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": max_tokens
        },
        "messages": messages_transformed,
        "tools": tools
    }
    
    logger.info("Формирование запроса завершено, отправка запроса в Yandex GPT.")
    
    return await _send_request_safety(payload, yandex_api_key)

async def generate_yandex_embeddings_response(text: str, model: str, yandex_api_key: str, folder_id: str) -> tuple[TextEmbeddingResponse, None] | tuple[None, dict]:
    """
    Генерация эмбеддингов с использованием Yandex GPT.

    :param text: Текст для генерации эмбеддингов.
    :param model: Модель для генерации эмбеддингов.
    :param yandex_api_key: API ключ для авторизации.
    :param folder_id: ID папки для Yandex API.
    :return: Вектор эмбеддинга или сообщение об ошибке.
    """
    logger.debug("Начало генерации эмбеддингов.")
    
    async def _send_request_safety(payload: dict, api_key: str) -> tuple[TextEmbeddingResponse, None] | tuple[None, dict]:
        """
        Отправка запроса в Yandex GPT с обработкой ошибок.

        :param payload: Данные запроса.
        :param api_key: API ключ для авторизации.
        :return: Вектор эмбеддинга или сообщение об ошибке.
        """
        url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key.startswith('t1') else f"Api-Key {api_key}",
            'x-folder-id': folder_id
        }

        logger.debug(f"Отправка запроса на {url} с заголовками: {headers} и данными: {payload}")
        
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            response = await client.post(url, headers=headers, json=payload, timeout=30)
            elapsed_time = time.time() - start_time
            logger.debug(f"Время выполнения запроса: {elapsed_time:.2f} секунд")

            if response.status_code == 200:
                logger.debug(f"Ответ от Yandex GPT: {response.json()}")
                await increment_yandex_metric_counter("yandexgpt_embeddings_requests", labels={"model": model})
                return TextEmbeddingResponse(**response.json()), None
            else:
                logger.error(f"Ошибка при выполнении запроса: {response.text}, статус код: {response.status_code}")
                return None, {
                    "error": {
                        "message": f"Ошибка: {response.text}",
                        "type": "api_error",
                        "param": None,
                        "code": response.status_code
                    }
                }
    
    model_uri = _get_embedding_model_uri(model, folder_id)
    logger.debug(f"Определён URI модели: {model_uri}")
    
    payload = {
        "modelUri": model_uri,
        "text": text
    }
    
    logger.info("Формирование запроса завершено, отправка запроса в Yandex GPT.")
    
    return await _send_request_safety(payload, yandex_api_key)


    

def _get_completions_model_uri(model: str, folder_id: str) -> str:
    if model == "gpt-4o":
        return f"gpt://{folder_id}/yandexgpt/latest"
    elif model == "gpt-4o-mini":
        return f"gpt://{folder_id}/yandexgpt-lite/latest"
    elif model.startswith(("gpt://", "ds://")):
        return model
    else:
        return f"gpt://{folder_id}/{model}"
    
def _get_embedding_model_uri(model: str, folder_id: str) -> str:
    if model in ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]:
        return f"emb://{folder_id}/text-search-doc/latest"
    elif model.startswith(("emb://", "ds://")):
        return model
    else:
        return f"emb://{folder_id}/{model}"
    
def _adapt_messages_for_yandexgpt(messages: list[dict], tools: list[dict] | None = None) -> tuple[list[dict], dict]:
    logger.debug(f"Начало преобразования сообщений в формат Yandex GPT. Сообщения: {messages}, инструменты: {tools}")
    messages_transformed = []
    
    # TODO, WORKAROUND
    # Создаем defaultdict, где каждый элемент - это deque
    called_functions = {}
    
    i = -1
    while i + 1 < len(messages):
        i += 1
        message = messages[i]
        # если tool, преобразовать в assistant с toolResultList
        # если ассистент с tool_calls/function_call, преобразовать в assistant с toolCallList
        # во всех остальных случаях только контент на текст поменять
        
        if message.get('role') == 'tool':
            name = called_functions.get(message.get('tool_call_id'))
            toolResults = [
                    {"functionResult": {
                        "name": name,
                        "result": message.get('content')
                    }}
                ]
            
            while i + 1 < len(messages) and messages[i + 1].get('role') == 'tool':
                i += 1
                message = messages[i]

                name = called_functions.get(message.get('tool_call_id'))
                toolResults.append({
                    "functionResult": {
                        "name": name,
                        "result": message.get('content')
                    }
                })
            
            messages_transformed.append({
                "role": "assistant",
                "toolResultList": {
                    "toolResults": toolResults
                }
            })
        elif message.get('role') == 'assistant' and message.get('tool_calls'):
            # внутри tool_calls список с  лежит id, type, function(name, arguments)
            toolCalls = []
            for tool_call in message.get('tool_calls'):
                try: 
                    name = tool_call.get('function').get('name')
                    arguments = json.loads(tool_call.get('function').get('arguments'))
                except:
                    logger.error(f"Ошибка при извлечении name и arguments из tool_call: {tool_call}")
                    continue
                toolCalls.append({
                    "functionCall": {
                        "name": name,
                        "arguments": arguments
                    }
                })
                called_functions[tool_call.get('id')] = name
                
                
            messages_transformed.append({
                "role": "assistant",
                "toolCallList": {
                    "toolCalls": toolCalls
                }
            })
        else:
            # system, user, assistant w/o tool_calls or tool_results
            role = message.get('role')
            content = message.get('content', None)
            
            if content:
                # EXPERIMENTAL
                if not isinstance(content, str):
                    content = str(content)
                
            messages_transformed.append({
                "role": role,
                "text": content
            })
        
    logger.debug(f"Преобразование сообщений в формат Yandex GPT завершено, результат: {messages_transformed}, вызовы инструментов: {called_functions}")
    return messages_transformed, called_functions

def _adapt_message_for_openai(yandex_response: CompletionResponse, model: str) -> dict:
            
    tool_calls_bool = yandex_response.alternatives[0].status == "ALTERNATIVE_STATUS_TOOL_CALLS"
    if tool_calls_bool:
        tool_calls_obj = [
            {
                "id": "call_" + ''.join(random.choices(string.ascii_letters + string.digits, k=8)),
                #"id": "call_abc123",
                "type": "function",
                "function": {
                    "name": item.functionCall.name,
                    "arguments": json.dumps(item.functionCall.arguments)
                }
            } for item in yandex_response.alternatives[0].message.toolCallList.toolCalls
        ]
        
        role = "assistant"
        finish_reason = "tool_calls"
        content = None
    else:
        role = "assistant"
        tool_calls_obj = None
        finish_reason = "stop"
        content = yandex_response.alternatives[0].message.text
    
    # Формирование ответа в формате OpenAI
    openai_format_response = {
        "id": "chatcmpl-42",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"{model}-by-{yandex_response.modelVersion}",
        "system_fingerprint": "42",
        "choices": [{
            "index": 0,
            "message": {
                "role": role,
                "content": content,
                "tool_calls": tool_calls_obj
            },
            "logprobs": None,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": yandex_response.usage.inputTextTokens,
            "completion_tokens": yandex_response.usage.completionTokens,
            "total_tokens": yandex_response.usage.totalTokens
        }
    }

    logger.debug(f"Формирование ответа в формате OpenAI завершено, ответ: {openai_format_response}")
    return openai_format_response


def _adapt_message_for_openai_stream_tool_calls(yandex_response: CompletionResponse, model: str) -> dict:
    tool_calls_obj: list = [
        {
            "id": "call_" + ''.join(random.choices(string.ascii_letters + string.digits, k=8)),
            #"id": "call_abc123",
            "type": "function",
            "function": {
                "name": item.functionCall.name,
                "arguments": json.dumps(item.functionCall.arguments)
            }
        } for item in yandex_response.alternatives[0].message.toolCallList.toolCalls
    ]
    
    openai_format_response = {
        "id": "chatcmpl-42",  # Здесь можно использовать уникальный ID
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-4o-mini-2024-07-18",
        "system_fingerprint": "fp_e2bde53e6",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",  # Добавляем роль
                    "content": None,
                    "tool_calls": tool_calls_obj
                },
                "logprobs": None,
                "finish_reason": "tool_calls"
            }
        ]
    }
    
    logger.debug(f"Формирование ответа в формате OpenAI завершено, ответ: {openai_format_response}")
    return openai_format_response

def _tool_from_message(messsage):
    pass