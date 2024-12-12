import logging
from pythonjsonlogger import jsonlogger
import dotenv
import os
import re

dotenv.load_dotenv()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

def obfuscate_message(message: str):
    """Obfuscate sensitive information."""
    result = re.sub(r'(Api-Key|Bearer|OAuth|OPENAI_API_KEY:|Ключ:|ключ:) [A-Za-z0-9_\-@]+', "***API_KEY_OBFUSCATED***", message)
    return result

class YcLoggingFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        record.message = obfuscate_message(record.getMessage())
        
        super(YcLoggingFormatter, self).add_fields(log_record, record, message_dict)
        log_record['logger'] = record.name
        log_record['level'] = str.replace(str.replace(record.levelname, "WARNING", "WARN"), "CRITICAL", "FATAL")

ycLogHandler = logging.StreamHandler()
ycLogHandler.setFormatter(YcLoggingFormatter('%(message)s %(level)s %(logger)s'))
