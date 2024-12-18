# pip install langcain langchain-openai

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

FOLDER_ID = os.getenv("FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
PROXY_URL = "https://o2y.ai-cookbook.ru"

base_url = f"{PROXY_URL}/v1"
api_key = f"{FOLDER_ID}@{YANDEX_API_KEY}"

llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="yandexgpt/latest",
)
print(llm.invoke("Hello, world!"))