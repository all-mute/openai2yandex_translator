# Используем официальный образ Python
FROM python:3.12-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы в контейнер
COPY . /app/

# Устанавливаем FastAPI и Uvicorn
RUN pip install -r requirements.txt

EXPOSE 9041

RUN cp .env.example .env

# healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:9041/health || exit 1

# Команда для запуска приложения
CMD ["gunicorn", "main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:9041"]
