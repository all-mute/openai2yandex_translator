from loguru import logger
from app.yc_log_handler import ycLogHandler

# Удаляем все существующие обработчики
logger.remove()

# Добавляем ycLogHandler
logger.add(ycLogHandler, level="INFO")

# Экспортируем настроенный logger
__all__ = ["logger"] 