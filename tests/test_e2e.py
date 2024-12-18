import pytest
import openai
import time, json, os
from dotenv import load_dotenv
from loguru import logger

logger.add("logs/test.log")

load_dotenv('.testenv')

FOLDER_ID = os.getenv("FOLDER_ID", "")
API_KEY = os.getenv("YANDEX_API_KEY", "")
#PROXY_URL = "https://d5det46m4e43042pnnfj.apigw.yandexcloud.net"
PROXY_URL = "http://localhost:9041"
#PROXY_URL = "https://bbafv6hdkrihhcvh9u78.containers.yandexcloud.net"

system_prompt = "Answer with only one word to my question"
user_prompt = "What is the meaning of life?"
emb_prompt = "Hello Yandex!"
ds_model_id = "bt120qtlha5a2aisl2ih"

# Configure the OpenAI client to use the proxy server
oai = openai.Client(api_key=f"{FOLDER_ID}@{API_KEY}", base_url=f"{PROXY_URL}/v1/")

@pytest.mark.parametrize("system_prompt, user_prompt, model", [
    (system_prompt, user_prompt, "gpt-4o"),
    (system_prompt, user_prompt, "gpt-4o-mini"),
    (system_prompt, user_prompt, "yandexgpt/latest"),
    (system_prompt, user_prompt, "yandexgpt-lite/latest"),
    (system_prompt, user_prompt, f"gpt://{FOLDER_ID}/yandexgpt/latest"),
    (system_prompt, user_prompt, f"gpt://{FOLDER_ID}/yandexgpt-lite/latest"),
    #(system_prompt, user_prompt, f"ds://{ds_model_id}"),
])
def test_completion_with_alternative_model(system_prompt, user_prompt, model):
    time.sleep(0.25)
    retries = 3
    
    for _ in range(retries):  # Попробуем выполнить запрос до 3 раз
        response = oai.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model=model,
        )
        
        if response and hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            if content is not None and content != "" and isinstance(content, str):
                break  # Успешный ответ, выходим из цикла
    assert content is not None and content != "" and isinstance(content, str)
    
@pytest.mark.parametrize("model", [
    "gpt-4o",
    "gpt-4o-mini",
    "yandexgpt/latest",
    "yandexgpt-lite/latest",
    f"gpt://{FOLDER_ID}/yandexgpt/latest",
    f"gpt://{FOLDER_ID}/yandexgpt-lite/latest",
    #f"ds://{ds_model_id}",
])
def test_streaming_completion(model):
    time.sleep(0.5)  # Allow some time for the server to be ready

    response = oai.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        stream=True,
    )
    
    collected_chunks = []
    collected_messages = []
    
    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk.choices[0].delta.content  # extract the message
        collected_messages.append(chunk_message)

    collected_messages = [m for m in collected_messages if m is not None]
    full_reply_content = ''.join(collected_messages)
    assert full_reply_content is not None and full_reply_content != "" and isinstance(full_reply_content, str)
    
@pytest.mark.parametrize("text, model", [
    (emb_prompt, "text-search-doc/latest"),
    (emb_prompt, "text-search-query/latest"),
    (emb_prompt, "text-embedding-3-large"),
    (emb_prompt, "text-embedding-3-small"),
    (emb_prompt, f"emb://{FOLDER_ID}/text-search-doc/latest"),
    (emb_prompt, f"emb://{FOLDER_ID}/text-search-query/latest"),
])
def test_embeddings_with_alternative_model(text, model):
    response = oai.embeddings.create(input = [text], model=model)
    
    vector = response.data[0].embedding
    assert len(vector) > 0 and isinstance(vector, list)
    assert isinstance(vector[0], float)
    
@pytest.mark.parametrize("text, model", [
    (emb_prompt, "text-search-doc/latest"),
    (emb_prompt, "text-search-query/latest")
])
def test_embeddings_batch_with_alternative_model(text, model):
    n = 33
    retries = 2
    for attempt in range(retries):
        response = oai.embeddings.create(input=[text] * n, model=model)
        if response and hasattr(response, 'data') and len(response.data) == n:
            break
    else:
        pytest.fail("Не удалось получить корректный ответ после нескольких попыток.")
    
    vector = response.data[0].embedding
    assert len(vector) > 0 and isinstance(vector, list)
    assert isinstance(vector[0], float)


def test_completion_with_invalid_model():
    try:
        response = oai.chat.completions.create(
                model="invalid-model",
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
    except Exception as e:
        assert e.status_code == 404
        assert isinstance(e, openai.APIStatusError)
        

def test_completion_with_unvalid_parameters():
    try:
        response = oai.chat.completions.create(
                model=1000,
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
    except Exception as e:
        assert e.status_code == 500
        #assert isinstance(e, openai.APIStatusError)


def test_completion_with_invalid_parameters():
    try:
        response = oai.chat.completions.create(
                model="yandexgpt/latest",
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=2,
            max_tokens=10_000,
        )
    except Exception as e:
        assert e.status_code == 422
        assert isinstance(e, openai.APIStatusError)
        

def test_completion_with_additional_parameters():

    response = oai.chat.completions.create(
            model="yandexgpt/latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            presence_penalty=2,
            seed=42,
        )
    
    assert not (hasattr(response, 'error') and response.error)
    
def test_completion_with_empty_message():
    try:
        response = oai.chat.completions.create(
                model="yandexgpt/latest",
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ""}  # Пустое сообщение
            ]
        )
        raise
    except Exception as e:
        assert e.status_code == 400  # Ожидаем ошибку из-за пустого сообщения


def test_completion_with_long_message():
    long_message = "Привет " * 100000  # Очень длинное сообщение
    try:
        response = oai.chat.completions.create(
                model="yandexgpt/latest",
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": long_message}
            ]
        )
        raise
    except Exception as e:
        assert e.status_code == 400  # Ожидаем ошибку из-за слишком длинного сообщения


def test_completion_with_correct_parameters():
    response = oai.chat.completions.create(
            model="yandexgpt/latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=100,
        )
    assert response and hasattr(response, 'choices') and response.choices
    content = response.choices[0].message.content
    assert content is not None and content != "" and isinstance(content, str)
    
def test_completion_with_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Получить текущую погоду в указанном городе",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    response = oai.chat.completions.create(
        model="yandexgpt/latest",
        messages=[
            {"role": "system", "content": "Вы - помощник, который может узнавать погоду"},
            {"role": "user", "content": "Какая погода в Москве?"}
        ],
        tools=tools,
        temperature=0,
        #tool_choice="auto"
    )

    assert response and hasattr(response, 'choices') and response.choices
    choice = response.choices[0]
    
    # Проверяем, что модель запросила использование функции
    assert choice.message.tool_calls is not None
    assert len(choice.message.tool_calls) > 0
    
    tool_call = choice.message.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    
    # Проверяем параметры вызова функции
    function_args = json.loads(tool_call.function.arguments)
    assert "city" in function_args
    assert function_args["city"] == "Москва"
    

    
def test_completion_with_correct_parameters_with_max_temp():
    response = oai.chat.completions.create(
            model="yandexgpt/latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1,
            max_tokens=100,
        )
    assert response and hasattr(response, 'choices') and response.choices
    content = response.choices[0].message.content
    assert content is not None and content != "" and isinstance(content, str)
    


@pytest.mark.skip(reason="no way of currently testing this")
def test_embeddings_with_invalid_parameters():

    response = oai.embeddings.create(input = [emb_prompt], model='invalid-model')
    
    assert hasattr(response, 'error') and response.error
    
    
def test_streaming_completion_with_tools_e2e():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Получить текущую погоду в указанном городе",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    response: list[openai.ChatCompletionChunk] = oai.chat.completions.create(
        model="yandexgpt/latest",
        messages=[
            {"role": "system", "content": "Вы - помощник, который может узнавать погоду"},
            {"role": "user", "content": "Какая погода в Москве?"}
        ],
        tools=tools,
        temperature=0,
        stream=True
    )

    collected_chunks = []
    collected_tool_calls = []

    # Собираем все чанки
    for chunk in response:
        logger.info(chunk)
        collected_chunks.append(chunk)
        if chunk.choices[0]:
            if chunk.choices[0].delta:
                if chunk.choices[0].delta.tool_calls:
                    collected_tool_calls.extend(chunk.choices[0].delta.tool_calls)
            else:
                if chunk.choices[0].message and chunk.choices[0].message.get('tool_calls'):
                    collected_tool_calls.extend(chunk.choices[0].message.get('tool_calls'))
            

    # Проверяем что получили только один чанк
    assert len(collected_chunks) == 1, "Должен быть получен только один чанк с tool_calls"
    
    # Проверяем что finish_reason именно tool_calls
    assert collected_chunks[0].choices[0].finish_reason == "tool_calls"
    
    # Проверяем что получили ровно один tool_call
    assert len(collected_tool_calls) == 1, "Должен быть получен ровно один tool_call"
    
    # Проверяем содержимое tool_call
    tool_call = collected_tool_calls[0]
    assert tool_call.get('function').get('name') == "get_weather"
    
    # Проверяем параметры вызова функции
    function_args = json.loads(tool_call.get('function').get('arguments'))
    assert "city" in function_args
    assert function_args["city"].lower() == "москва"

    # Проверяем что content пустой, так как это tool_call
    assert collected_chunks[0].choices[0].delta is None
    
    
def test_completion_with_multiple_tool_calls_and_responses():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Получить текущую погоду в указанном городе",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города"
                        }
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Получить текущее время в указанном городе",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    response = oai.chat.completions.create(
        model="yandexgpt/latest",
        messages=[
            {"role": "system", "content": "Вы - помощник, который может узнавать погоду и время"},
            {"role": "user", "content": "Какая погода и время в Москве?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Москва"}'
                        }
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"city": "Москва"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": "Температура 20°C, солнечно",
                "tool_call_id": "call_1"
            },
            {
                "role": "tool",
                "content": "Текущее время: 14:00",
                "tool_call_id": "call_2"
            }
        ],
        tools=tools,
        temperature=0
    )

    assert response and hasattr(response, 'choices') and response.choices
    content = response.choices[0].message.content
    assert content is not None and content != "" and isinstance(content, str)
    assert "20" in content and "14" in content

def test_completion_with_multiple_tool_calls_and_responses_tools_are_the_same():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Получить текущую погоду в указанном городе",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города"
                        }
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Получить текущее время в указанном городе",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    response = oai.chat.completions.create(
        model="yandexgpt/latest",
        messages=[
            {"role": "system", "content": "Вы - помощник, который может узнавать погоду и время"},
            {"role": "user", "content": "Какая погода в Москве?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Москва"}'
                        }
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Москва"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": "Температура 20°C, солнечно",
                "tool_call_id": "call_1"
            },
            {
                "role": "tool",
                "content": "Температура 22°C, солнечно",
                "tool_call_id": "call_2"
            }
        ],
        tools=tools,
        temperature=0
    )

    assert response and hasattr(response, 'choices') and response.choices
    content = response.choices[0].message.content
    assert content is not None and content != "" and isinstance(content, str)
    assert "20" in content or "22" in content

def test_streaming_completion_with_tools_e2e_hard():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Получить текущую погоду в указанном городе",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города"
                        }
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Получить текущее время в указанном городе",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    response = oai.chat.completions.create(
        model="yandexgpt/latest",
        messages=[
            {"role": "system", "content": "Вы - помощник, который может узнавать погоду и время"},
            {"role": "user", "content": "Какая погода в Москве?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Москва"}'
                        }
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"city": "Москва"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": "Температура 20°C, солнечно",
                "tool_call_id": "call_1"
            },
            {
                "role": "tool",
                "content": "Текущее время: 14:00",
                "tool_call_id": "call_2"
            }
        ],
        tools=tools,
        temperature=0,
        stream=True
    )

    collected_chunks = []
    collected_messages = []
    
    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk.choices[0].delta.content  # extract the message
        collected_messages.append(chunk_message)

    collected_messages = [m for m in collected_messages if m is not None]
    full_reply_content = ''.join(collected_messages)
    assert full_reply_content is not None and full_reply_content != "" and isinstance(full_reply_content, str)