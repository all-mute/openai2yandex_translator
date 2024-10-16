from fastapi.responses import RedirectResponse, StreamingResponse
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from app.yandex import generate_yandexgpt_response, generate_yandex_embeddings_response, generate_yandexgpt_stream_response
from app.models import CompletionResponse, TextEmbeddingResponse
import os, sys, time, json
from loguru import logger
from dotenv import load_dotenv
import asyncio

load_dotenv()

# 1. Если вы найдете scheme запроса к openai, пожалуйста, дайте знать tg @nongilgameshj
# 2. Перед тем как придираться к отсутствию валидации и сериализации, пожалуйста, ответьте на вопрос: "А зачем оно в прокси?"

# Загрузка конфига
with open('config.json', 'r') as f:
    config = json.load(f)

# Загрузка ключей для автоавторизации
autoauth_keys = tuple(config.get("autoauth_keys", []))

# Настройка для батч-обработки эмбеддингов
embeddings_retry_num = config.get("embeddings_retry_num", 3)
s_embeddings_delays = config.get("s_embeddings_delays", [0.050, 0.500, 3])
embeddings_batch_size = config.get("embeddings_batch_size", 5)

# Уровень логирования
LOG_LEVEL = config.get("log_level", "INFO")

# Проверяем, запущено ли приложение на Vercel
is_vercel = os.getenv("VERCEL", False)

# Настраиваем логирование
if is_vercel:
    # Логи выводятся в консоль
    logger.add(sys.stdout, format="{time} {level} {message}", level=LOG_LEVEL)
else:
    # Логи записываются в файл
    logger.add("logs/debug.log", format="{time} {level} {message}", level=LOG_LEVEL, rotation="100 MB")


app = FastAPI(logger=logger)

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
        if openai_api_key in autoauth_keys:
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
        stream = body.get("stream", False)
        
        logger.debug(f"Полученные данные: model={model}, max_tokens={max_tokens}, temperature={temperature}, messages={messages}, stream={stream}")
        logger.info(f"Используемая модель: {model}")
        
        # Генерация ответа от Yandex GPT
        if stream:
            
            return StreamingResponse(generate_yandexgpt_stream_response(messages, model, temperature, max_tokens, yandex_api_key, folder_id), media_type="text/event-stream")
        else:
            yandex_response, yandex_error = await generate_yandexgpt_response(messages, model, temperature, max_tokens, yandex_api_key, folder_id)
        
            if yandex_error:
                logger.error(f"Ошибка при генерации ответа от Yandex GPT: {yandex_error}")
                return yandex_error
            
            # Формирование ответа в формате OpenAI
            openai_format_response = {
                "id": "chatcmpl-42",
                "object": "chat.completion",
                "created": str(time.time()),
                "model": f"{model}-by-{yandex_response.modelVersion}",
                "system_fingerprint": "42",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": yandex_response.alternatives[0].message.text,
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": yandex_response.usage.inputTextTokens,
                    "completion_tokens": yandex_response.usage.completionTokens,
                    "total_tokens": yandex_response.usage.totalTokens
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
        
        if openai_api_key in autoauth_keys:
            yandex_api_key, folder_id = YANDEX_API_KEY, FOLDER_ID
            logger.debug("Используются ключи по умолчанию YANDEX_API_KEY и FOLDER_ID.")
        else:
            folder_id, yandex_api_key = openai_api_key.split("@")
            logger.debug(f"Используемые ключи: folder_id={folder_id}, yandex_api_key={yandex_api_key}")
        
        body = await request.json()
        model = body.get("model")
        input_data = body.get("input", [None])  # Обработка случая, если input отсутствует
        logger.debug(f"Полученные данные: model={model}, input_data={input_data}, len(input_data)={len(input_data)}")
        logger.info(f"Используемая модель: {model}")
        
        if len(input_data) > 100:
            logger.warning(f"Enormous len(input_data) > 100: len={len(input_data)}")
        
        """if len(input_data) == 1:
            input_data = input_data[0]
            yandex_vector, yandex_error = await generate_yandex_embeddings_response(input_data, model, yandex_api_key, folder_id)
            yandex_vectors = [yandex_vector]
        else:
            yandex_vectors, yandex_error = await generate_yandex_embeddings_response_batch(input_data, model, yandex_api_key, folder_id, embeddings_retry_num, embeddings_batch_size)"""
        
        yandex_vectors, yandex_error = await generate_yandex_embeddings_response_batch(input_data, model, yandex_api_key, folder_id, embeddings_retry_num, embeddings_batch_size)
        
        if yandex_error:
            logger.error(f"Ошибка при генерации эмбеддинга от Yandex GPT: {yandex_error}")
            return yandex_error
        
        data_list = []
        for i, item in enumerate(yandex_vectors):
            data_list.append({
                "object": "embedding",
                "index": i,
                "embedding": item.embedding,
            })
        
        sumNumTokens = sum(int(item.numTokens) for item in yandex_vectors)
        
        # Формирование ответа в формате OpenAI
        openai_format_response = {
            "object": "list",
            "data": data_list,
            "model": f"{model}-by-{yandex_vectors[0].modelVersion}",
            "usage": {
                "prompt_tokens": sumNumTokens,
                "total_tokens": sumNumTokens,
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

@app.get("/badge")
def get_badge():
    return RedirectResponse("https://img.shields.io/badge/status-online-brightgreen.svg")

async def generate_yandex_embeddings_response_batch(arr: list, model, yandex_api_key, folder_id, retry_num, batch_size: int):
    logger.info(f"Начинаем обработку массива из {len(arr)} текстов с размером батча {batch_size}.")
    
    text_kv = [(index, text) for index, text in enumerate(arr)]
    logger.debug(f"Преобразованный массив текстов: {text_kv}")
    
    batch_results = {}
    
    for i in range(0, len(text_kv), batch_size):
        batch = text_kv[i:i + batch_size]
        logger.debug(f"Обработка батча {i // batch_size + 1}: {batch}")
        
        tasks = [process_safely_one_embedding(index, text, model, yandex_api_key, folder_id, retry_num) for index, text in batch]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug(f"Результаты обработки батча {i // batch_size + 1}: {results}")
        
        for j, yandex_vector, yandex_error in results:
            if yandex_error is not None:
                logger.error(f"Ошибка при обработке {j} текста {arr[j]}: {yandex_error}")
                return None, yandex_error
            else:
                logger.debug(f"Успешно обработан текст {arr[j]} с индексом {j}.")
                batch_results[j] = yandex_vector
    
    result_array = [None] * len(arr)
    for index, vector in batch_results.items():
        result_array[index] = vector
    
    logger.info("Обработка всех текстов завершена. Возвращаем массив векторов.")
    return result_array, None
    
async def process_safely_one_embedding(index, input_data, model, yandex_api_key, folder_id, retry_num):
    retries = 0
    while retries < retry_num:
        yandex_vector, yandex_error = await generate_yandex_embeddings_response(input_data, model, yandex_api_key, folder_id)
        if yandex_error is None:
            return index, yandex_vector, None
        else:
            
            time.sleep(s_embeddings_delays[retries])
            retries += 1
            
            logger.warning(f"Ошибка при обработке текста {input_data}: {yandex_error}. Попытка {retries}/{retry_num} после ожидания.")
            if retries == retry_num:
                logger.error(f"Все попытки завершились неудачей для текста {input_data}.")
                return index, None, yandex_error


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9041, reload=True, log_level="debug")