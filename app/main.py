import uvicorn
from fastapi import FastAPI, HTTPException, Request
from app.yandex import generate_yandexgpt_response, generate_yandex_embeddings_response
import os, sys, time
from loguru import logger

# Проверяем, запущено ли приложение на Vercel
is_vercel = os.getenv("VERCEL", False)

# Настраиваем логирование
if is_vercel:
    # Логи выводятся в консоль
    logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
else:
    # Логи записываются в файл
    logger.add("logs/debug.log", format="{time} {level} {message}", level="INFO", rotation="100 MB")


app = FastAPI()

# Получение переменных окружения
FOLDER_ID = os.getenv("FOLDER_ID", "")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")

@app.post("/v1/chat/completions")
async def completion(request: Request):
    logger.info("Обработка запроса на генерацию ответа.")
    logger.debug(f"Запрос: {request.method} {request.url}")
    logger.debug(f"Заголовки: {request.headers}")
    logger.debug(f"Тело запроса: {await request.json()}")
    logger.debug(f"IP-адрес отправителя: {request.client.host}")
    try:
        logger.debug("Получение OPENAI_API_KEY из заголовка запроса.")
        openai_api_key = request.headers.get("Authorization", "").split("Bearer ")[-1].strip()
        
        logger.debug(f"Извлеченный OPENAI_API_KEY: {openai_api_key}")
        
        # Определение Yandex API ключа и ID папки
        if openai_api_key in ("sk-my", "", " "):
            yandex_api_key, folder_id = YANDEX_API_KEY, FOLDER_ID
            logger.debug("Использование Yandex API ключа из переменных окружения.")
        else:
            folder_id, yandex_api_key = openai_api_key.split("@")
            logger.debug(f"Использование Yandex API ключа: {yandex_api_key} и ID папки: {folder_id}.")
        
        # Получение данных из запроса
        body = await request.json()
        model = body.get("model")
        max_tokens = body.get("max_tokens", 2048)
        temperature = body.get("temperature", 0.3)
        messages = body.get("messages", [])
        
        logger.debug(f"Полученные данные: model={model}, max_tokens={max_tokens}, temperature={temperature}, messages={messages}")
        logger.info(f"Используемая модель: {model}")
        
        # Генерация ответа от Yandex GPT
        yandex_response, yandex_error = generate_yandexgpt_response(messages, model, temperature, max_tokens, yandex_api_key, folder_id)
        
        if yandex_error:
            logger.error(f"Ошибка при генерации ответа от Yandex GPT: {yandex_error}")
            return yandex_error
        
        # Формирование ответа в формате OpenAI
        openai_format_response = {
            "id": "chatcmpl-42",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
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
        
        logger.debug("Формирование ответа в формате OpenAI завершено.")
        return openai_format_response
    except HTTPException as e:
        logger.error(f"HTTP ошибка: {str(e)}")
        return {"error": str(e)}  # Возврат ошибки в формате JSON
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}")
        return {"error": "An unexpected error occurred."}  # Обработка неожиданных ошибок

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    logger.info("Обработка запроса на генерацию эмбеддинга.")
    logger.debug(f"Запрос: {request.method} {request.url}")
    logger.debug(f"Заголовки: {request.headers}")
    logger.debug(f"Тело запроса: {await request.json()}")
    logger.debug(f"IP-адрес отправителя: {request.client.host}")
    try:
        # Извлечение OPENAI_API_KEY из заголовка "Authorization: Bearer $OPENAI_API_KEY"
        openai_api_key = request.headers.get("Authorization", "").split("Bearer ")[-1].strip()
        logger.debug(f"Извлечённый OPENAI_API_KEY: {openai_api_key}")
        
        if openai_api_key in ("sk-my", "", " "):
            yandex_api_key, folder_id = YANDEX_API_KEY, FOLDER_ID
            logger.debug("Используются ключи по умолчанию YANDEX_API_KEY и FOLDER_ID.")
        else:
            folder_id, yandex_api_key = openai_api_key.split("@")
            logger.debug(f"Используемые ключи: folder_id={folder_id}, yandex_api_key={yandex_api_key}")
        
        body = await request.json()
        model = body.get("model")
        input_data = body.get("input", [None])[0]  # Обработка случая, если input отсутствует
        logger.debug(f"Полученные данные: model={model}, input_data={input_data}")
        logger.info(f"Используемая модель: {model}")
        
        yandex_vector = generate_yandex_embeddings_response(input_data, model, yandex_api_key, folder_id)
        
        # Формирование ответа в формате OpenAI
        openai_format_response = {
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": yandex_vector,
            }],
            "model": model,
            "usage": {
                "prompt_tokens": 42,  # Примерное значение
                "total_tokens": 42
            }
        }
        
        logger.info("Формирование ответа в формате OpenAI завершено.")
        return openai_format_response
    except HTTPException as e:
        logger.error(f"HTTP ошибка: {str(e)}")
        return {"error": str(e)}  # Возврат ошибки в формате JSON
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}")
        return {"error": "An unexpected error occurred."}  # Обработкаunexpected ошибок

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