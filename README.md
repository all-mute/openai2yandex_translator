# Yandex GPT Translator

Это лаконичное fastapi приложение (прокси), которое транслирует запросы OpenAI<->Yandex Cloud Foundational Models, чтобы сервисы Yandex Cloud можно было использовать в сторонних фреймворках через OpenAI SDK. 

Например:

```python
import openai

# С аутентификацией запроса
client = openai.Client(api_key=f"{FOLDER_ID}@{API_KEY_OR_IAM_KEY}", base_url=f"{proxy_url}/v1/")
# Или с автоматической аутентификацией
client = openai.Client(api_key=f"sk-my", base_url=f"{proxy_url}/v1/")
```

## Функционал:

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
        # или f'gpt://{FOLDER_ID}/yandexgpt/latest' 
        # или f'ds://{MODEL_ID}'
        max_tokens=2000,
        temperature=0.1,
    )
    
# эмбеддинги текста
client.embeddings.create(input = ['В каком году был основан Яндекс?'], model='text-search-doc/latest').data[0].embedding # или model=f'emb://{FOLDER_ID}/text-search-doc/latest'
```

Подробные примеры содержатся в файле `test.py`.

### Поддерживаются:

* Все модели генерации текста, uri которых начинаются с `gpt://`
* Все Embedding модели, uri которых начинаются с `emb://`
* Все дообученные модели генерации текста, uri которых начинаются с `ds://`

### Аутентификация

* Если опенаи ключ в запросе указать `sk-my`, ` ` или пустым, то кредиты для работы с Yandex Cloud будут искаться в переменных окружения. Задайте их для работы с этим сценарием в `.env` или `Dockerfile`.

* Если в данную проксю будет ходит несколько пользователей, то в качестве OpenAI ключа указывайте folder_id и статический апи-ключ или IAM-ключ, разделяя их символом `@` (например `folder_id@iam_key`).

## Запуск

Быстрый запуск на vercel:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com%2Fvercel%2Fnext.js%2Ftree%2Fcanary%2Fexamples%2Fhello-world)

1. Если вам нужен доступ к ресурсам с автоматической аутентификацией, заполните данные параметры (in `.env` file, Dockerfile or cloud environment):
    - `FOLDER_ID`: your Yandex Cloud folder id
    - `YANDEX_API_KEY`: your Yandex Cloud API key
2. Запустите приложение:
    - `docker-compose up -d --build` 
    - или (команды аналогичны для `podman`) `docker build -t image_name .`, затем `docker run -d -p 127.0.0.1:8000:8000 --name container_name image_name` 
    - или (для локального тестирования) `pip install -r requirements.txt`, затем `python main.py`

## Решение проблем

Если у вас возникли проблемы по работе с этим приложением, пожалуйста, создайте issue в этом репозитории, он активно поддерживается.

* Чтобы ходить в дообученную gpt, пользователь/сервисный аккаунт должны быть участниками проекта DataShpere с ролью `developer`
