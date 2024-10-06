import requests
import time
from fastapi import HTTPException
from loguru import logger

def generate_yandexgpt_response(messages, model: str, temperature: float, max_tokens: int, yandex_api_key: str, folder_id: str) -> str:
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
    
    def send_request_safety(payload: dict, api_key: str) -> tuple:
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
        
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        if response.status_code == 200:
            response_json = response.json()
            text = response_json['result']['alternatives'][0]['message']['text']
            logger.info("Запрос успешно выполнен, получен ответ от Yandex GPT.")
            return text, None
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
        message['text'] = message.pop('content', '')

    # Определение URI модели
    model_uri = (
        f"gpt://{folder_id}/yandexgpt/latest" if model == "gpt-4o" else
        f"gpt://{folder_id}/yandexgpt-lite/latest" if model == "gpt-4o-mini" else
        model if model.startswith(("gpt://", "ds://")) else
        f"gpt://{folder_id}/{model}"
    )
    
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
    return send_request_safety(payload, yandex_api_key)

def generate_yandex_embeddings_response(text: str, model: str, yandex_api_key: str, folder_id: str) -> list:
    """
    Генерация эмбеддингов с использованием Yandex GPT.

    :param text: Текст для генерации эмбеддингов.
    :param model: Модель для генерации эмбеддингов.
    :param yandex_api_key: API ключ для авторизации.
    :param folder_id: ID папки для Yandex API.
    :return: Вектор эмбеддинга или сообщение об ошибке.
    """
    logger.debug("Начало генерации эмбеддингов.")
    
    def send_request_safety(payload: dict, api_key: str) -> list:
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
        
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        if response.status_code == 200:
            logger.info("Запрос успешно выполнен, получен ответ от Yandex GPT.")
            return response.json().get("embedding", [])
        else:
            logger.error(f"Ошибка при выполнении запроса: {response.text}, статус код: {response.status_code}")
            return [f"Ошибка: {response.status_code}, {response.text}"]
    
    # Формирование URI модели
    model_uri = (
        f"emb://{folder_id}/text-search-doc/latest" if model in ["text-embedding-3-large", "text-embedding-3-small"] else
        model if model.startswith(("emb://", "ds://")) else
        f"emb://{folder_id}/{model}"
    )
    
    logger.debug(f"Определён URI модели: {model_uri}")
    
    payload = {
        "modelUri": model_uri,
        "text": text
    }
    
    logger.info("Формирование запроса завершено, отправка запроса в Yandex GPT.")
    return send_request_safety(payload, yandex_api_key)