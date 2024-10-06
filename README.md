# Yandex GPT Translator

- [Функционал](#функционал)
    - [Поддерживаемые модели](#поддерживаемые-модели)
    - [Аутентификация](#аутентификация)
    - [Использование моделей gpt-4 как YandexGPT](#использование-моделей-gpt-4-как-yandexgpt)
    - [Планы](#планы)
- [Деплой](#деплой)
    - [Коммунальные трансляторы (без деплоя)](#коммунальные-трансляторы-без-деплоя)
    - [Быстрый запуск на vercel](#быстрый-запуск-на-vercel)
    - [Локальный/облачный запуск](#локальный-облачный-запуск)
- [Решение проблем](#решение-проблем)


Это лаконичное fastapi приложение (прокси), которое транслирует запросы OpenAI<->Yandex Cloud Foundational Models, чтобы сервисы Yandex Cloud можно было использовать в сторонних фреймворках через OpenAI SDK. 

Например:

```python
import openai

# С аутентификацией запроса
client = openai.Client(api_key=f"{FOLDER_ID}@{API_KEY_OR_IAM_KEY}", base_url=f"{proxy_url}/v1/")
# Или с автоматической аутентификацией
client = openai.Client(api_key=f"sk-my", base_url=f"{proxy_url}/v1/")
```

*Вы можете использовать SDK на любом языке, в том числе js, go, и т.д.*

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

Подробные примеры содержатся в файлах [example.py](examples/example.py) и [example.js](examples/example.js).

### Поддерживаемые модели:

* Все модели генерации текста, uri которых начинаются с `gpt://`
* Все Embedding модели, uri которых начинаются с `emb://`
* Все дообученные модели генерации текста, uri которых начинаются с `ds://`
* Все дообученные Embedding модели, uri которых начинаются с `ds://` (пока такого функционала нет в Yandex Cloud)

Классификаторы, function calling, logprobs не поддерживаются.

### Аутентификация

* **На стороне прокси.** Если опенаи ключ в запросе указать `sk-my`, ` ` или пустым, то кредиты для работы с Yandex Cloud будут искаться в переменных окружения. Задайте их для работы с этим сценарием в `.env` или `Dockerfile`.

* **На стороне пользователя.** Если в данную проксю будет ходит несколько пользователей, то в качестве OpenAI ключа указывайте folder_id и статический апи-ключ или IAM-ключ, разделяя их символом `@` (например `folder_id@iam_key`).

### Использование моделей gpt-4 как YandexGPT

Если вы ограничены в выборе моделей, но можете указать хотя-бы api-ключ и base_url, то можете обращаться к моделям OpenAI - запросы будут мапиться на модели YandexGPT:
- gpt-4o -> yandexgpt/latest
- gpt-4o-mini -> yandexgpt-lite/latest
- text-embedding-3-large -> text-search-doc/latest
- text-embedding-3-small -> text-search-doc/latest

### Планы

* Стриминг
* Ошибки
* Добавить поддержку классификаторов
* Добавить поддержку дообученных эмбеддингов
* Добавить поддержку YandexART
* Добавить поддержку 

## Деплой

### Коммунальныые трансляторы (без деплоя)

1. **Yandex Cloud**, `ru-central1-a`. Stateless режим, запросы не логируются. http://openai-api-translator.llmplay.ru:8000

2. **Vercel** https://openai2yandex-translator.vercel.app 


### Быстрый запуск на vercel:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fall-mute%2Fyagpt2openai_translator)

Для использования автоматической аутентификации, заполните `FOLDER_ID` & `YANDEX_API_KEY` на странице деплоя.

### Локальный/облачный запуск

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
* При деплое через serverless платформы (vervel, yc functions) не забудьте выставить timeout 30 секунд
