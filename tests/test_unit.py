import pytest
from app.yandex.completions import _adapt_messages
from app.yandex.models import Message as YaChatCompletionMessage
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from fastapi import HTTPException
from app.yandex.completions import _get_completions_model_uri

def test_adapt_messages():
    # Создаем фиктивные данные для теста
    messages = [
        ChatCompletionSystemMessageParam(role='system', content='You are a helpful assistant.'),
        ChatCompletionUserMessageParam(role='user', content='Hello!'),
        ChatCompletionAssistantMessageParam(role='assistant', content='Hi there!'),
        ChatCompletionAssistantMessageParam(role='assistant', tool_calls=[
            {'id': '123', 'function': {'name': 'test_function', 'arguments': '{"arg1": "value1"}'}}
        ]),
        ChatCompletionToolMessageParam(role='tool', tool_call_id='123', content='Tool result')
    ]

    # Ожидаемый результат
    expected_result = [
        YaChatCompletionMessage(**{"role": "system", "text": "You are a helpful assistant."}),
        YaChatCompletionMessage(**{"role": "user", "text": "Hello!"}),
        YaChatCompletionMessage(**{"role": "assistant", "text": "Hi there!"}),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolCallList": {
                "toolCalls": [
                    {"functionCall": {"name": "test_function", "arguments": {"arg1": "value1"}}}
                ]
            }
        }),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolResultList": {
                "toolResults": [
                    {"functionResult": {"name": "test_function", "content": "Tool result"}}
                ]
            }
        }),
    ]

    # Запускаем тестируемую функцию
    result = _adapt_messages(messages)

    # Проверяем, что результат соответствует ожиданиям
    assert result == expected_result 

def test_adapt_messages_no_tools():
    # Тест без вызова инструментов
    messages = [
        ChatCompletionSystemMessageParam(role='system', content='You are a helpful assistant.'),
        ChatCompletionUserMessageParam(role='user', content='Hello!'),
        ChatCompletionAssistantMessageParam(role='assistant', content='Hi there!'),
    ]

    expected_result = [
        YaChatCompletionMessage(**{"role": "system", "text": "You are a helpful assistant."}),
        YaChatCompletionMessage(**{"role": "user", "text": "Hello!"}),
        YaChatCompletionMessage(**{"role": "assistant", "text": "Hi there!"}),
    ]

    result = _adapt_messages(messages)
    assert result == expected_result

def test_adapt_messages_sequential_tool_calls():
    # Тест с последовательным вызовом инструментов 
    messages = [
        ChatCompletionSystemMessageParam(role='system', content='You are a helpful assistant.'),
        ChatCompletionUserMessageParam(role='user', content='Hello!'),
        ChatCompletionAssistantMessageParam(role='assistant', content='Hi there!'),
        ChatCompletionAssistantMessageParam(role='assistant', tool_calls=[
            {'id': '123', 'function': {'name': 'test_function', 'arguments': '{"arg1": "value1"}'}}
        ]),
        ChatCompletionToolMessageParam(role='tool', tool_call_id='123', content='Tool result'),
        ChatCompletionAssistantMessageParam(role='assistant', tool_calls=[
            {'id': '333', 'function': {'name': 'test_function', 'arguments': '{"arg1": "value2"}'}}
        ]),
        ChatCompletionToolMessageParam(role='tool', tool_call_id='333', content='Tool result 2')
    ]

    # Ожидаемый результат
    expected_result = [
        YaChatCompletionMessage(**{"role": "system", "text": "You are a helpful assistant."}),
        YaChatCompletionMessage(**{"role": "user", "text": "Hello!"}),
        YaChatCompletionMessage(**{"role": "assistant", "text": "Hi there!"}),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolCallList": {
                "toolCalls": [
                    {"functionCall": {"name": "test_function", "arguments": {"arg1": "value1"}}}
                ]
            }
        }),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolResultList": {
                "toolResults": [
                    {"functionResult": {"name": "test_function", "content": "Tool result"}}
                ]
            }
        }),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolCallList": {
                "toolCalls": [
                    {"functionCall": {"name": "test_function", "arguments": {"arg1": "value2"}}}
                ]
            }
        }),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolResultList": {
                "toolResults": [
                    {"functionResult": {"name": "test_function", "content": "Tool result 2"}}
                ]
            }
        }),
    ]

    # Запускаем тестируемую функцию
    result = _adapt_messages(messages)

    # Проверяем, что результат соответствует ожиданиям
    assert result == expected_result 
    
def test_adapt_messages_sequential_tool_calls_diff_names():
    # Тест с последовательным вызовом инструментов 
    messages = [
        ChatCompletionSystemMessageParam(role='system', content='You are a helpful assistant.'),
        ChatCompletionUserMessageParam(role='user', content='Hello!'),
        ChatCompletionAssistantMessageParam(role='assistant', content='Hi there!'),
        ChatCompletionAssistantMessageParam(role='assistant', tool_calls=[
            {'id': '123', 'function': {'name': 'test_function', 'arguments': '{"arg1": "value1"}'}}
        ]),
        ChatCompletionToolMessageParam(role='tool', tool_call_id='123', content='Tool result'),
        ChatCompletionAssistantMessageParam(role='assistant', tool_calls=[
            {'id': '333', 'function': {'name': 'TEST_function', 'arguments': '{"arg1": "value2"}'}}
        ]),
        ChatCompletionToolMessageParam(role='tool', tool_call_id='333', content='Tool result 2')
    ]

    # Ожидаемый результат
    expected_result = [
        YaChatCompletionMessage(**{"role": "system", "text": "You are a helpful assistant."}),
        YaChatCompletionMessage(**{"role": "user", "text": "Hello!"}),
        YaChatCompletionMessage(**{"role": "assistant", "text": "Hi there!"}),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolCallList": {
                "toolCalls": [
                    {"functionCall": {"name": "test_function", "arguments": {"arg1": "value1"}}}
                ]
            }
        }),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolResultList": {
                "toolResults": [
                    {"functionResult": {"name": "test_function", "content": "Tool result"}}
                ]
            }
        }),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolCallList": {
                "toolCalls": [
                    {"functionCall": {"name": "TEST_function", "arguments": {"arg1": "value2"}}}
                ]
            }
        }),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolResultList": {
                "toolResults": [
                    {"functionResult": {"name": "TEST_function", "content": "Tool result 2"}}
                ]
            }
        }),
    ]

    # Запускаем тестируемую функцию
    result = _adapt_messages(messages)

    # Проверяем, что результат соответствует ожиданиям
    assert result == expected_result 

def test_adapt_messages_parallel_tool_calls():
    # Тест с параллельным вызовом инструментов
    messages = [
        ChatCompletionSystemMessageParam(role='system', content='You are a helpful assistant.'),
        ChatCompletionUserMessageParam(role='user', content='Hello!'),
        ChatCompletionAssistantMessageParam(role='assistant', content='Hi there!'),
        ChatCompletionAssistantMessageParam(role='assistant', tool_calls=[
            {'id': '123', 'function': {'name': 'test_function', 'arguments': '{"arg1": "value1"}'}},
            {'id': '111', 'function': {'name': 'test_function22', 'arguments': '{"arg1": "value3"}'}}
        ]),
        ChatCompletionToolMessageParam(role='tool', tool_call_id='123', content='Tool result'),
        ChatCompletionToolMessageParam(role='tool', tool_call_id='111', content='Tool result 3'),
        ChatCompletionAssistantMessageParam(role='assistant', tool_calls=[
            {'id': '333', 'function': {'name': 'TEST_function', 'arguments': '{"arg1": "value2"}'}}
        ]),
        ChatCompletionToolMessageParam(role='tool', tool_call_id='333', content='Tool result 2')
    ]

    # Ожидаемый результат
    expected_result = [
        YaChatCompletionMessage(**{"role": "system", "text": "You are a helpful assistant."}),
        YaChatCompletionMessage(**{"role": "user", "text": "Hello!"}),
        YaChatCompletionMessage(**{"role": "assistant", "text": "Hi there!"}),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolCallList": {
                "toolCalls": [
                    {"functionCall": {"name": "test_function", "arguments": {"arg1": "value1"}}},
                    {"functionCall": {"name": "test_function22", "arguments": {"arg1": "value3"}}}
                ]
            }
        }),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolResultList": {
                "toolResults": [
                    {"functionResult": {"name": "test_function", "content": "Tool result"}},
                    {"functionResult": {"name": "test_function22", "content": "Tool result 3"}}
                ]
            }
        }),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolCallList": {
                "toolCalls": [
                    {"functionCall": {"name": "TEST_function", "arguments": {"arg1": "value2"}}}
                ]
            }
        }),
        YaChatCompletionMessage(**{
            "role": "assistant",
            "toolResultList": {
                "toolResults": [
                    {"functionResult": {"name": "TEST_function", "content": "Tool result 2"}}
                ]
            }
        }),
    ]

    # Запускаем тестируемую функцию
    result = _adapt_messages(messages)

    # Проверяем, что результат соответствует ожиданиям
    assert result == expected_result 
    


@pytest.mark.parametrize("model, folder_id, expected_uri", [
    ("yandexgpt/latest", "folder123", "gpt://folder123/yandexgpt/latest"),
    ("gpt-4o", "folder123", "gpt://folder123/yandexgpt/latest"),
    ("gpt-3.5", "folder456", "gpt://folder456/yandexgpt/latest"),
    ("gpt://folder789/custom_model/latest", "folder789", "gpt://folder789/custom_model/latest"),
    ("ds://folder000/custom_model/rc", "folder000", "ds://folder000/custom_model/rc"),
])
def test_get_completions_model_uri(model, folder_id, expected_uri):
    assert _get_completions_model_uri(model, folder_id) == expected_uri
    
