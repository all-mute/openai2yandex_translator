import pytest
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

@pytest.fixture
def llm_setup():
    load_dotenv('.testenv')
    
    FOLDER = os.getenv("FOLDER_ID")
    API_KEY = os.getenv("YANDEX_API_KEY")
    
    OPENAI_API_KEY = f"{FOLDER}@{API_KEY}"
    base_url = f"http://localhost:9041/v1/"
    
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=base_url,
        model="yandexgpt/latest",
        temperature=0
    )
    return llm

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

def test_simple_addition(llm_setup):
    messages = [HumanMessage("Сколько будет 2 + 2?")]
    ai_msg = llm_setup.invoke(messages)
    assert isinstance(ai_msg.content, str)
    assert "4" in ai_msg.content

def test_calculation_flow(llm_setup):
    tools = [add, multiply]
    llm_with_tools = llm_setup.bind_tools(tools)
    messages = []
    
    # Первый запрос: 3 * 12
    messages.append(HumanMessage("What is 3 * 12? Use tools!"))
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    
    messages.append(llm_with_tools.invoke(messages))
    
    # Второй запрос: 11 + 49
    messages.append(HumanMessage("What is 11 + 49? Use tools!"))
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    
    messages.append(llm_with_tools.invoke(messages))
    
    # Сложение результатов
    messages.append(HumanMessage("Теперь сложи результаты!"))
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    
    final_message = llm_with_tools.invoke(messages)
    messages.append(final_message)
    
    assert "96" in final_message.content