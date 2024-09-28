# Yandex GPT Translator

Это лаконичное fastapi приложение (прокси), которое транслирует запросы OpenAI<->Yandex Cloud, чтобы сервисы Yandex Cloud можно было использовать в сторонних фреймворках через OpenAI SDK. Например:

```python
import openai

# С аутентификацией запроса
client = openai.Client(api_key=f"{FOLDER_ID}@{API_KEY_OR_IAM_KEY}", base_url=f"{proxy_url}/v1/")
# Или без
client = openai.Client(api_key=f"sk-my", base_url=f"{proxy_url}/v1/")
```

Функционал:

```python
# генерация текста
client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": 'В каком году был основан Яндекс?',
            }
        ],
        model="yandexgpt/latest",
        max_tokens=2000,
        temperature=0.1,
    )
    
# эмбеддинги текста
client.embeddings.create(input = ['В каком году был основан Яндекс?'], model='text-search-doc/latest').data[0].embedding
```

Подробные примеры содержаттся в файле `test.py`.

## Поддерживаются:

* Все GPT модели, uri которых начинаются с `gpt://`
* Все Embedding модели, uri которых начинаются с `emb://`

## Аутентификация

* Если опенаи ключ в запросе указать `sk-my`, ` ` или пустым, то кредиты для работы с Yandex Cloud будут искаться в переменных окружения. Задайте их для работы с этим сценарием в `.env` или `Dockerfile`.

* Если в данную проксю будет ходит несколько пользователей, то в качестве OpenAI ключа указывайте folder_id и статический апи-ключ или IAM-ключ, разделяя их символом `@` (например `folder_id@iam_key`).

## Запуск

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com%2Fvercel%2Fnext.js%2Ftree%2Fcanary%2Fexamples%2Fhello-world)

1. Если вам нужен доступ к ресурсам с аутентификацией по-умолчанию, заполните данные параметры (in `.env` file, Dockerfile or cloud environment):
    - `FOLDER_ID`: your Yandex Cloud folder id
    - `YANDEX_API_KEY`: your Yandex Cloud API key
2. Запустите приложение (команды аналогичны для `podman`):
    `docker-compose up -d --build` или `docker build -t image_name .`, `docker run -d -p 127.0.0.1:8000:8000 --name container_name image_name` или (для локального тестирования) `uvicorn main:app --host 0.0.0.0 --port 8000` 
