import httpx
import time, json
from fastapi import HTTPException
from loguru import logger
from app.models import CompletionResponse, TextEmbeddingResponse, CompletionRequest, TextEmbeddingRequest

async def generate_yandexgpt_stream_response(messages, model: str, temperature: float, max_tokens: int, yandex_api_key: str, folder_id: str):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {yandex_api_key}" if yandex_api_key.startswith('t1') else f"Api-Key {yandex_api_key}",
        'x-folder-id': folder_id
    }
    
    logger.debug("Начинаем преобразование сообщений в формат Yandex GPT.")
    # Преобразование сообщений в формат Yandex GPT
    for message in messages:
        message['text'] = message.get('content')
        message.pop('content', None)  # Удаление поля 'content', если оно существует
        logger.debug(f"Преобразовано сообщение: {message}")

    model_uri = _get_completions_model_uri(model, folder_id)
    logger.debug(f"Определён URI модели: {model_uri}")
    
    payload = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": True,
            "temperature": temperature,
            "maxTokens": max_tokens
        },
        "messages": messages
    }
    
    logger.info("Формирование запроса завершено, отправка запроса в Yandex GPT.")
    
    generated_len = 0
    finished = False
    
    logger.info("Отправка запроса в Yandex GPT...")
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=15)
        logger.debug(f"Ответ от Yandex GPT получен с кодом статуса: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Ошибка при отправке запроса: {response.text}")
            return
        
        async for line in response.aiter_lines():
            if line:
                logger.debug(f"Полученная строка: {line}")
                json_line = json.loads(line)  # Преобразование строки в JSON
                
                if 'result' in json_line:
                    response_obj = CompletionResponse(**json_line['result'])
                    logger.debug(f"Преобразованный объект: {response_obj}")

                    content = response_obj.alternatives[0].message.text[generated_len:]
                    generated_len = len(response_obj.alternatives[0].message.text)

                    response_data = {
                        "id": "chatcmpl-42",  # Здесь можно использовать уникальный ID
                        "object": "chat.completion.chunk",
                        "created": str(time.time()),
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
                    break

    if finished:
        logger.info("Завершение передачи данных.")
        yield "data: [DONE]\n\n"

                    

async def generate_yandexgpt_response(messages, model: str, temperature: float, max_tokens: int, yandex_api_key: str, folder_id: str) -> tuple[CompletionResponse, None] | tuple[None, dict]:
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

        :param payload: Данные запроса.
        :param api_key: API ключ для авторизации.
        :return: Ответ от Yandex GPT или ошибка.
        """
        
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
    
    # Преобразование сообщений в формат Yandex GPT
    for message in messages:
        # Извлечение текста из поля 'content' и замена его на 'text'
        message['text'] = message.get('content')
        message.pop('content')  # Удаление поля 'content', если оно существует

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
        "messages": messages
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
        
        response = httpx.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            
            logger.info("Запрос успешно выполнен, получен ответ от Yandex GPT.")
            logger.debug(f"Ответ от Yandex GPT: {response.json()}")
            
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
    
