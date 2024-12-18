# OpenAI to Yandex GPT API Adapter

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
![Test Status](https://github.com/ai-cookbook/openai-yandexgpt-adapter/actions/workflows/docker-image.yml/badge.svg)
![Test Status](https://github.com/ai-cookbook/openai-yandexgpt-adapter/actions/workflows/python-app.yml/badge.svg)
![Vercel](https://vercelbadge.vercel.app/api/all-mute/openai-yandexgpt-adapter)

**Use Yandex Cloud models from anywhere!**

**[Полная документация: ai-cookbook.ru/docs/adapter](https://ai-cookbook.ru/docs/adapter/)**

Данное приложение преобразует API-запросы формата OpenAI в запросы формата Yandex Cloud Foundational Models, что позволяет использовать Yandex Cloud Foundational Models через OpenAI SDK, lite-LLM, langchain, других dev библиотеках а также готовых пользовательских приложениях.

Рекомендуемый openai base_url: `https://o2y.ai-cookbook.ru/v1` ![badge](https://o2y.ai-cookbook.ru/badge)

## Быстрый старт:

```python
import openai

client = openai.Client(api_key=f"{FOLDER_ID}@{API_KEY_OR_IAM_KEY}", base_url="https://o2y.ai-cookbook.ru/v1")
```

*Вы можете использовать SDK на любом языке, в том числе js, go, и т.д.*

Примеры: [python OpenAI SDK](./examples/example.py), [js](./examples/example.js), [langchain](./examples/langchain-example.py)

## Решение проблем

Если у вас возникли проблемы при работе с этим приложением, **пожалуйста, создайте issue** в этом репозитории, он активно поддерживается. Оперативно по проблемам писать tg `@nongilgameshj`


### Дисклеймер

Данный проект не является официальным продуктом Yandex Cloud. Поддерживается командой ai-cookbook.ru.
