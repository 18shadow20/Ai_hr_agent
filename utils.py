import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import json
import re
import uuid
from openai import OpenAI
from database import Message

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

openai_client = OpenAI(api_key=GOOGLE_API_KEY)



if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY не установлен в переменных окружения")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY не установлен в переменных окружения")

try:
    pinecone_api = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    raise ValueError(f"Ошибка при инициализации Pinecone: {str(e)}")

index_name = "resume"
if index_name not in [i['name'] for i in pinecone_api.list_indexes()]:
    pinecone_api.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pinecone_api.Index(index_name)
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

def get_embedding(user_data):
    if isinstance(user_data, dict):
        text = " ".join([str(v) for v in user_data.values()])
    else:
        text = str(user_data)

    return model.encode(text).tolist()


def search_resume(query, top_k:int = 1, threshold: float= 0.5):
    """Сематический поиск"""
    if isinstance(query, dict):
        query_text = query.get("query", "")
    else:
        query_text = str(query)

    query_embedding = model.encode(query_text).tolist()
    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    matches = [i.to_dict() for i in result.get("matches", []) if i["score"] >= threshold]
    return matches


def question_ai(text):
    """Возвращает информацию о резюме кандидата"""
    completion = openai_client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": f"""
                 Проанализируй резюме и верни результат **строго в формате JSON**.
                 Никакого текста кроме JSON не пиши. Формат ответа:
                 {{
                  "name": "фамилия Имя кандидата",
                   "skills": ["Навык1", "Навык2", "Навык3"]
                 }}
                 Резюме:
                 {text}
                """
            }
        ],
    )
    response = completion.choices[0].message.content.strip()
    return response


def clear_text(user):
    cleaned = re.sub(r"```json|```", "", user).strip().replace("'", '"')
    try:
        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Модель вернула невалидный JSON:\n{user}")



def save_to_pinecone(user_data, metadata = None):
    embedding = get_embedding(user_data)
    vector_id = str(uuid.uuid4())
    index.upsert([
        { "id":vector_id,
          "values":embedding,
          "metadata": {"text":user_data}
          }
    ])
    return vector_id

def save_message(db, conversation_id: uuid.UUID, role: str, content: str):
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg