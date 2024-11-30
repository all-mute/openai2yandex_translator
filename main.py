from fastapi import FastAPI
from app.app import app
import os, sys, json
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Загрузка конфига
with open('config.json', 'r') as f:
    config = json.load(f)

# Уровень логирования
LOG_LEVEL = config.get("log_level", "INFO")

# Проверяем, запущено ли приложение на Vercel
is_vercel = os.getenv("VERCEL", False)

# Настраиваем логирование
if is_vercel:
    # Логи выводятся в консоль
    logger.add(sys.stdout, serialize=True, level=LOG_LEVEL)
else:
    # Логи записываются в файл
    logger.add("logs/debug.log", format="{time} {level} {message}", level=LOG_LEVEL, rotation="100 MB")

main_app = FastAPI(logger=logger)

main_app.include_router(app)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:main_app", host="0.0.0.0", port=9041, reload=True, log_level="info")
