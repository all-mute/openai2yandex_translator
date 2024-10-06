import sys
sys.path.append("../")

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from yandex import generate_yandexgpt_response, generate_yandex_embeddings_response
import time
import os
from loguru import logger

logger.add("debug.log", format="{time} {level} {message}", level="DEBUG", rotation="10 MB")

app = FastAPI()

# Получение переменных окружения
FOLDER_ID = os.getenv("FOLDER_ID", "")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")

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
        
        yandex_response, yandex_error = generate_yandexgpt_response(request.messages, request.model, request.temperature, request.max_tokens, yandex_api_key, folder_id)
        
        if yandex_error:
            return yandex_error
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
        return e
    except Exception as e:
        #print(e)
        return e

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

@app.get("/readyz")
def readiness_probe():
    # Здесь можно добавить проверки готовности
    return {"status": "ready"}

@app.get("/livez")
def liveness_probe():
    # Здесь можно добавить проверки работоспособности
    return {"status": "alive"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")