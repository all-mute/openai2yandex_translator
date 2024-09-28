import openai

FOLDER_ID = ""
API_KEY_OR_IAM_KEY = ""
#key = f"{FOLDER_ID}@{API_KEY_OR_IAM_KEY}"
key = "sk-my"
proxy_url = "http://0.0.0.0:8000"
oai = openai.Client(api_key=key, base_url=f"{proxy_url}/v1/")

def generate_text_oai(system_prompt, user_prompt, max_tokens=2000, temperature=0.1, model="yandexgpt/latest"):
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

def get_embedding(text, model):
   return oai.embeddings.create(input = [text], model=model).data[0].embedding

if __name__ == "__main__":
    print(generate_text_oai("You are a helpful assistant.", "What is the meaning of life? Answer in one word."))
    print(get_embedding("Hello Yandex!", "text-search-doc/latest")[:3], '...')