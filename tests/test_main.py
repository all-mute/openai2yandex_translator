import pytest
import openai
import os
import time

FOLDER_ID = os.getenv("FOLDER_ID", "")
API_KEY = os.getenv("YANDEX_API_KEY", "")
PROXY_URL = "http://localhost:8000"

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
    content = response.choices[0].message.content
    assert content is not None and content != "" and isinstance(content, str)

@pytest.mark.parametrize("text, model", [
    (emb_prompt, "text-search-doc/latest"),
    (emb_prompt, "text-search-query/latest"),
    (emb_prompt, f"emb://{FOLDER_ID}/text-search-doc/latest"),
    (emb_prompt, f"emb://{FOLDER_ID}/text-search-query/latest"),
])
def test_embeddings_with_alternative_model(text, model):
    response = oai.embeddings.create(input = [text], model=model)
    
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
    
