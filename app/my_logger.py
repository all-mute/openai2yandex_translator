from loguru import logger
from app.yc_log_handler import ycLogHandler
import os, sys, json
from dotenv import load_dotenv

load_dotenv()

# Уровень логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
log_type = os.getenv("LOG_TYPE", "volume")

# Настраиваем логирование
#logger.remove()
if log_type == "volume":
    # Логи записываются в файл
    logger.add("logs/debug.log", format="{time} {level} {message}", level=LOG_LEVEL, rotation="100 MB")
elif log_type == "vercel":
    # Логи выводятся в консоль
    logger.add(sys.stdout, format="{time} {level} {message}", level=LOG_LEVEL)
elif log_type == "yc":
    # Добавляем ycLogHandler
    logger.add(ycLogHandler, level=LOG_LEVEL)

# Экспортируем настроенный logger
__all__ = ["logger"] 