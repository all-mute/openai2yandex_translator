import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
import time
import os
from typing import List, Dict

app = FastAPI()

# Получение переменных окружения
folder_id = os.getenv("FOLDER_ID", "")
yandex_api_key = os.getenv("YANDEX_API_KEY", "")

# Функция для отправки запроса с повторной попыткой
def send_request_safety(history, api_key):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {yandex_api_key or api_key}",
    }

    response = requests.post(url, headers=headers, json=history)
    if response.status_code == 200:
        response_json = response.json()
        text = response_json['result']['alternatives'][0]['message']['text']
        return text
    elif response.status_code == 429:
        time.sleep(1)
        
        response = requests.post(url, headers=headers, json=history)
        if response.status_code == 200:
            response_json = response.json()
            text = response_json['result']['alternatives'][0]['message']['text']
            return text
        elif response.status_code == 429:
            time.sleep(3)
            response = requests.post(url, headers=headers, json=history)
            if response.status_code == 200:
                response_json = response.json()
                text = response_json['result']['alternatives'][0]['message']['text']
                return text
            else:
                return f"Error: {response.status_code}, {response.text}"
        else:
            return f"Error: {response.status_code}, {response.text}"
    else:
        return f"Error: {response.status_code}, {response.text}"

# Генерация ответа от Yandex GPT
def generate_yandexgpt_response(messages, model, temperature, max_tokens, openai_api_key) -> str:
    # Формирование запроса в формате Yandex GPT
    for message in messages:
        message['text'] = message['content']
        del message['content']
    
    # Формирование запроса в формате Yandex GPT
    payload = {
        "modelUri": f"gpt://{folder_id}/{model}",
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": max_tokens
        },
        "messages": messages
    }
    
    result = send_request_safety(payload, openai_api_key)
    
    # Задержка
    # time.sleep(0.05)
    
    return result

# Маршрут для обработки запроса
@app.post("/v1/chat/completions")
async def completion(request: Request):
    try:
        # take out OPENAI_API_KEY from header "Authorization: Bearer $OPENAI_API_KEY"
        openai_api_key = request.headers.get("Authorization", "").split("Bearer ")[-1].strip()
        
        body = await request.json()
        request.model = body.get("model")
        request.max_tokens = body.get("max_tokens", 2048)
        request.temperature = body.get("temperature", 0.1)
        request.messages = body.get("messages", [])
        
        yandex_response = generate_yandexgpt_response(request.messages, request.model, request.temperature, request.max_tokens, openai_api_key)
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
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)