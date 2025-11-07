import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

from database import SessionLocal
from utils import search_resume

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_candidates(query):
    """
    Ищет релевантных кандидатов и возвращает список кандидатов.
    """
    response = search_resume(query)
    if response:
        return "\n".join([f"- {c['metadata']['text']}" for c in response])
    return "Кандидаты не найдены."

tools = [
    Tool(
        name="GetCandidates",
        func=get_candidates,
        description=(
            "Используй этот инструмент, когда пользователь хочет найти кандидатов "
            "Возвращай только данные из базы, не придумывай."
        )
    )
]

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

llm = ChatOpenAI(
    model_name="gemini-2.5-flash",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key= GOOGLE_API_KEY,
    temperature=0.4,
    streaming=False,
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)









