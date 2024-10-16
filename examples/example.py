import openai

# укажите кредиты Yandex CLoud если используете проксю с включенной аутентификацией
# FOLDER_ID = ""
# API_KEY_OR_IAM_KEY = ""
# key = f"{FOLDER_ID}@{API_KEY_OR_IAM_KEY}"

# или оставьте ключ sk-my, если создали проксю с авоматической аутентификацией 
key = "sk-my"

# задайте адрес вашей прокси
proxy_url = "http://0.0.0.0:9041"

# создайте клиент OpenAI с измененным base_url
oai = openai.Client(api_key=key, base_url=f"{proxy_url}/v1/")

def generate_text_oai(system_prompt, user_prompt, max_tokens=2000, temperature=0.1, model=f"yandexgpt/latest"):
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
        #max_tokens=max_tokens,
        #temperature=0.1,
    )

    generated_text = response.choices[0].message.content
    return generated_text

def get_embedding(text, model=f"text-search-doc/latest"):
   return oai.embeddings.create(input = [text], model=model).data[0].embedding

def get_embedding_sync_batch(texts, model=f"text-search-doc/latest"):
   return oai.embeddings.create(input = texts, model=model).data

if __name__ == "__main__":
    # Поддерживаемые форматы моделей
    model = 'yandexgpt/latest'
    # или f'gpt://{FOLDER_ID}/yandexgpt/latest' 
    # или f'ds://{MODEL_ID}'
    # Для эмбеддингов 'text-search-doc/latest' 
    # или 'emb://{FOLDER_ID}/text-search-doc/latest'
    # или 'ds://{MODEL_ID}'
        
    print(generate_text_oai("You are a helpful assistant.", "What is the meaning of life? Answer in one word."))
    print(get_embedding("Hello Yandex!")[:3], '...')