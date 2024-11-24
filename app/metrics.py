import httpx
from loguru import logger
import time
from dotenv import load_dotenv
import os
import urllib.parse

load_dotenv()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
FOLDER_ID = os.getenv("FOLDER_ID")
USE_YANDEX_METRICS = os.getenv("USE_YANDEX_METRICS", "False").lower() == "true"

async def increment_yandex_metric_counter(metric_name: str, value: int = 1, labels: dict = None):
    """
    Увеличение счетчика метрики в Yandex Metrics.

    :param metric_name: Имя метрики.
    :param value: Значение для увеличения.
    :param labels: Метки для метрики в формате ключ:значение.
    :return: Ответ от Yandex Metrics или сообщение об ошибке.
    """
    if not USE_YANDEX_METRICS:
        logger.debug(f"Yandex Metrics не используются, пропускаем увеличение метрики {metric_name} на {value}.")
        return
    
    params = {
        "folderId": FOLDER_ID,
        "service": "custom"
    }
    
    query_params = urllib.parse.urlencode(params)
    
    url = f"https://monitoring.api.cloud.yandex.net/monitoring/v2/data/write?{query_params}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YANDEX_API_KEY}"
    }
    
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "labels": labels or {},
        "metrics": [
            {
                "name": metric_name,
                "type": "COUNTER",
                "value": value
            }
        ]
    }
    
    logger.debug(f"Отправка запроса на {url} с заголовками: {headers} и данными: {payload}")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            logger.info(f"Метрика {metric_name} успешно увеличена на {value}.")
            return response.json()
        else:
            logger.error(f"Ошибка при увеличении метрики: {response.text}, статус код: {response.status_code}")
            return {
                "error": {
                    "message": f"Ошибка: {response.text}",
                    "type": "api_error",
                    "param": None,
                    "code": response.status_code
                }
            }
