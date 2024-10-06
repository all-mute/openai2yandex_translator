import requests
import time
from fastapi import HTTPException

# Генерация ответа от Yandex GPT
def generate_yandexgpt_response(messages, model, temperature, max_tokens, yandex_api_key, folder_id) -> str:
    # Функция для отправки запроса в Yandex GPT
    def send_request_safety(payload, api_key):
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key.startswith('t1') else f"Api-Key {api_key}",
            'x-folder-id': f"{folder_id}"
        }

        response = requests.post(url, headers=headers, json=payload, timeout=180)
        if response.status_code == 200:
            response_json = response.json()
            text = response_json['result']['alternatives'][0]['message']['text']
            return text, None
        else:
            return None, {
                "error": {
                    "message": f"Ошибка: {response.text}",
                    "type": "api_error",
                    "param": None,
                    "code": response.status_code
                }
            }  # Это вызовет исключение, если статус-код ответа не 200
    
    # Формирование запроса в формате Yandex GPT
    for message in messages:
        message['text'] = message['content']
        del message['content']
        
    if model == "gpt-4o":
        model_uri = f"gpt://{folder_id}/yandexgpt/latest"
    elif model == "gpt-4o-mini":
        model_uri = f"gpt://{folder_id}/yandexgpt-lite/latest"
    elif model.startswith("gpt://") or model.startswith("ds://"):
        model_uri = model
    else:
        model_uri = f"gpt://{folder_id}/{model}"
    
    # Формирование запроса в формате Yandex GPT
    payload = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": max_tokens
        },
        "messages": messages
    }
    
    result = send_request_safety(payload, yandex_api_key)
    
    # Задержка от 429
    # time.sleep(0.05)
    
    return result

# Генерация ответа от Yandex GPT
def generate_yandex_embeddings_response(text, model, yandex_api_key, folder_id):
    # Функция для отправки запроса в Yandex GPT
    def send_request_safety(payload, api_key):
        url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key.startswith('t1') else f"Api-Key {api_key}",
            'x-folder-id': f"{folder_id}"
        }

        response = requests.post(url, headers=headers, json=payload, timeout=180)
        if response.status_code == 200:
            response_json = response.json()
            vector = response_json["embedding"]
            return vector
        else:
            return [f"Error: {response.status_code}, {response.text}"]
    
    # Формирование запроса в формате Yandex GPT
    if model == "text-embedding-3-large" or model == "text-embedding-3-small":
        model_uri = f"emb://{folder_id}/text-search-doc/latest"
    elif model.startswith("emb://") or model.startswith("ds://"):
        model_uri = model
    else:
        model_uri = f"emb://{folder_id}/{model}"
    
    payload = {
        "modelUri": model_uri,
        "text": text
    }
    
    result = send_request_safety(payload, yandex_api_key)
    
    # Задержка от 429
    # time.sleep(0.05)
    
    return result