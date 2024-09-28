import uvicorn
from fastapi import FastAPI, HTTPException, Request
import requests
import time
import os

import logging
logging.basicConfig(level=logging.DEBUG)


app = FastAPI()

# Получение переменных окружения
FOLDER_ID = os.getenv("FOLDER_ID", "")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")

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
            return text
        else:
            return f"Error: {response.status_code}, {response.text}"
    
    # Формирование запроса в формате Yandex GPT
    for message in messages:
        message['text'] = message['content']
        del message['content']
    
    # Формирование запроса в формате Yandex GPT
    payload = {
        "modelUri": f"{model}" if model.startswith("gpt://") or model.startswith("ds://") else f"gpt://{folder_id}/{model}",
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
def generate_yandex_embeddings_response(text, model, yandex_api_key, folder_id) -> str:
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
            return f"Error: {response.status_code}, {response.text}"
    
    # Формирование запроса в формате Yandex GPT
    payload = {
        "modelUri": f"{model}" if model.startswith("emb://") or model.startswith("ds://") else f"emb://{folder_id}/{model}",
        "text": text
    }
    
    result = send_request_safety(payload, yandex_api_key)
    
    # Задержка от 429
    # time.sleep(0.05)
    
    return result

# Маршрут для обработки запроса
@app.post("/v1/chat/completions")
async def completion(request: Request):
    try:
        # take out OPENAI_API_KEY from header "Authorization: Bearer $OPENAI_API_KEY"
        openai_api_key = request.headers.get("Authorization", "").split("Bearer ")[-1].strip()
        if openai_api_key in ("sk-my", "", " "):
            yandex_api_key, folder_id = YANDEX_API_KEY, FOLDER_ID
        else:
            folder_id, yandex_api_key = openai_api_key.split("@")
        
        body = await request.json()
        request.model = body.get("model")
        request.max_tokens = body.get("max_tokens", 2048)
        request.temperature = body.get("temperature", 0.3)
        request.messages = body.get("messages", [])
        
        yandex_response = generate_yandexgpt_response(request.messages, request.model, request.temperature, request.max_tokens, yandex_api_key, folder_id)
        # Формирование ответа в формате OpenAI
        openai_format_response = {
            "id": "chatcmpl-42",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "system_fingerprint": "42",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": yandex_response,
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 42,  # Примерное значение
                "completion_tokens": 42,
                "total_tokens": 42
            }
        }
        
        return openai_format_response
    except HTTPException as e:
        #raise HTTPException(status_code=500, detail=str(e))
        #print(e)
        pass
    except Exception as e:
        #print(e)
        pass

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    try:
        # take out OPENAI_API_KEY from header "Authorization: Bearer $OPENAI_API_KEY"
        openai_api_key = request.headers.get("Authorization", "").split("Bearer ")[-1].strip()
        if openai_api_key in ("sk-my", "", " "):
            yandex_api_key, folder_id = YANDEX_API_KEY, FOLDER_ID
        else:
            folder_id, yandex_api_key = openai_api_key.split("@")
        
        body = await request.json()
        request.model = body.get("model")
        request.input = body.get("input")[0]
        
        yandex_vector = generate_yandex_embeddings_response(request.input, request.model, yandex_api_key, folder_id)
        # Формирование ответа в формате OpenAI
        openai_format_response = {
            "object": "list",
            "data": [
                {
                "object": "embedding",
                "index": 0,
                "embedding": yandex_vector,
                }
            ],
            "model": request.model,
            "usage": {
                "prompt_tokens": 42,
                "total_tokens": 42
            }
        }
        
        return openai_format_response
    except HTTPException as e:
        #raise HTTPException(status_code=500, detail=str(e))
        #print(e)
        pass
    except Exception as e:
        #print(e)
        pass

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")