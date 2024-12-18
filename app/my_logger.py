from loguru import logger
from app.yandex.yc_log_handler import ycLogHandler
import os, sys, json
from dotenv import load_dotenv

load_dotenv()

# Уровень логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
log_type = os.getenv("LOG_TYPE", "stdout")

# Настраиваем логирование
if log_type == "dev":
    pass
else:
    logger.remove()

# Логи записываются в файл
if log_type == "volume":
    logger.add("logs/debug.log", format="{time} {level} {message}", level=LOG_LEVEL, rotation="100 MB")
    
# Логи выводятся в консоль
elif log_type == "stdout":  
    logger.add(sys.stdout, format="{time} {level} {message}", level=LOG_LEVEL)
    
# Логи отправляются в Yandex Cloud Logging
elif log_type == "yc":
    # Добавляем ycLogHandler
    logger.add(ycLogHandler, level=LOG_LEVEL) 

# Экспортируем настроенный logger
__all__ = ["logger"] 