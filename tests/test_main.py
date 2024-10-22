import pytest
import openai
import time, json, os
from dotenv import load_dotenv

load_dotenv()

FOLDER_ID = os.getenv("FOLDER_ID", "")
API_KEY = os.getenv("YANDEX_API_KEY", "")
PROXY_URL = "http://localhost:9041"

system_prompt = "Answer with only one word to my question"
user_prompt = "What is the meaning of life?"
emb_prompt = "Hello Yandex!"
ds_model_id = "bt12j06dfr7pmncjipab"

# Configure the OpenAI client to use the proxy server
oai = openai.Client(api_key=f"{FOLDER_ID}@{API_KEY}", base_url=f"{PROXY_URL}/v1/")

@pytest.mark.parametrize("system_prompt, user_prompt, model", [
    (system_prompt, user_prompt, "gpt-4o"),
    (system_prompt, user_prompt, "gpt-4o-mini"),
    (system_prompt, user_prompt, "yandexgpt/latest"),
    (system_prompt, user_prompt, "yandexgpt-lite/latest"),
    (system_prompt, user_prompt, f"gpt://{FOLDER_ID}/yandexgpt/latest"),
    (system_prompt, user_prompt, f"gpt://{FOLDER_ID}/yandexgpt-lite/latest"),
    (system_prompt, user_prompt, f"ds://{ds_model_id}"),
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
    f"ds://{ds_model_id}",
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

@pytest.mark.parametrize("key", [
    "sk-my",
    #"invalid-key",
    "invalid@key",
])
def test_completion_with_invalid_authorization(key):
    oai_wrong = openai.Client(api_key=key, base_url=f"{PROXY_URL}/v1/")

    response = oai_wrong.chat.completions.create(
            model="gpt://test-folder-id/alternative-model",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the meaning of life?"}
            ]
        )
    
    assert hasattr(response, 'error') and response.error


def test_completion_with_invalid_model():

    response = oai.chat.completions.create(
            model="invalid-model",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

    assert hasattr(response, 'error') and response.error


def test_completion_with_invalid_parameters():

    response = oai.chat.completions.create(
            model="yandexgpt/latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=2,
            max_tokens=10_000,
        )
    
    assert hasattr(response, 'error') and response.error
        

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

@pytest.mark.skip(reason="no way of currently testing this")
def test_embeddings_with_invalid_parameters():

    response = oai.embeddings.create(input = [emb_prompt], model='invalid-model')
    
    assert hasattr(response, 'error') and response.error
    
