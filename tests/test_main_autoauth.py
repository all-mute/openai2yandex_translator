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
oai = openai.Client(api_key=f"sk-my", base_url=f"{PROXY_URL}/v1/")

@pytest.mark.parametrize("system_prompt, user_prompt, model", [
    (system_prompt, user_prompt, "gpt-4o"),
    (system_prompt, user_prompt, "gpt-4o-mini"),
    (system_prompt, user_prompt, "yandexgpt/latest"),
    (system_prompt, user_prompt, "yandexgpt-lite/latest"),
    #(system_prompt, user_prompt, f"gpt://{FOLDER_ID}/yandexgpt/latest"),
    #(system_prompt, user_prompt, f"gpt://{FOLDER_ID}/yandexgpt-lite/latest"),
    (system_prompt, user_prompt, f"ds://{ds_model_id}"),
])
def test_completion_with_alternative_model(system_prompt, user_prompt, model):
    time.sleep(0.25)
    
    for _ in range(3):  # Попробуем выполнить запрос до 3 раз
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

@pytest.mark.parametrize("text, model", [
    (emb_prompt, "text-search-doc/latest"),
    (emb_prompt, "text-search-query/latest"),
    #(emb_prompt, f"emb://{FOLDER_ID}/text-search-doc/latest"),
    #(emb_prompt, f"emb://{FOLDER_ID}/text-search-query/latest"),
])
def test_embeddings_with_alternative_model(text, model):
    response = oai.embeddings.create(input = [text], model=model)
    
    vector = response.data[0].embedding
    assert len(vector) > 0 and isinstance(vector, list)
    assert isinstance(vector[0], float)
