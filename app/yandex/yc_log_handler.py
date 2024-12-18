import logging
from pythonjsonlogger import jsonlogger
import dotenv
import os
import re

dotenv.load_dotenv()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

def obfuscate_message(message: str):
    """Obfuscate sensitive information."""
    result = re.sub(r'(Api-key|Api-Key|Bearer|OAuth|OPENAI_API_KEY:|Ключ:|ключ:) [A-Za-z0-9_\-@]+', "***API_KEY_OBFUSCATED***", message)
    return result

class YcLoggingFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        record.message = obfuscate_message(record.getMessage())
        
        super(YcLoggingFormatter, self).add_fields(log_record, record, message_dict)
        log_record['logger'] = record.name
        log_level = record.levelname
        if log_level == "WARNING":
            log_level = "WARN"
        elif log_level == "CRITICAL":
            log_level = "FATAL"
        elif log_level == "SUCCESS":
            log_level = "INFO"
        log_record['level'] = log_level

ycLogHandler = logging.StreamHandler()
ycLogHandler.setFormatter(YcLoggingFormatter('%(message)s %(level)s %(logger)s'))
