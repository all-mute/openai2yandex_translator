# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы в контейнер
COPY . /app/

# Устанавливаем FastAPI и Uvicorn
RUN pip install -r requirements.txt

EXPOSE 8000

ENV FOLDER_ID=""
ENV YANDEX_API_KEY=""

# Команда для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
