from fastapi import FastAPI, UploadFile, Depends, Request, HTTPException, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.responses import FileResponse
import os
import uuid
from openai import OpenAI
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select
from database import User, SessionLocal
from utils import question_ai, clear_text, save_to_pinecone, save_message, search_resume
from chat import agent
import whisper
import tempfile
import io

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

openai_client = OpenAI(api_key=GOOGLE_API_KEY)

whisper_model = None
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("base")
    return whisper_model


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally: db.close()

app = FastAPI()


@app.post("/upload")
async def upload_file(file: UploadFile, db: Session = Depends(get_db)):
    text = extract_text(file.file)
    user_text = question_ai(text)
    user_json = clear_text(user_text)
    save_to_pinecone(text)
    user = User(name=user_json["name"], skills=user_json["skills"])
    st = select(User).where(User.name == user_json["name"])
    exist_user = db.scalar(st)
    if exist_user:
        return {"message": "Пользователь уже есть в базе"}
    db.add(user)
    db.commit()
    db.refresh(user)
    return user






templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})


@app.post("/api/chat")
async def chat(request:Request, db: Session = Depends(get_db)):
    conversation_id = uuid.uuid4()

    data = await request.json()
    message = data.get("message")
    response = agent.invoke({"input": message})

    save_message(db, conversation_id, role="user", content=message)
    save_message(db, conversation_id, role="assistant", content=response["output"])
    return JSONResponse({"response": response["output"]})



class SearchUser(BaseModel):
    query: str


@app.post("/search")
async def search_user(query: SearchUser):
    response = search_resume(query.query)
    return {"candidates":response}


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), db: Session = Depends(get_db)):
    """Транскрибирует аудио и отправляет в LLM"""
    tmp_file_path = None
    try:
        content_type = audio.content_type or "audio/webm"
        extension_map = {
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav"
        }
        extension = extension_map.get(content_type, ".webm")

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        model = get_whisper_model()
        result = model.transcribe(tmp_file_path, language="ru")
        transcribed_text = result["text"]

        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        if not transcribed_text.strip():
            return JSONResponse({"error": "Не удалось распознать речь"}, status_code=400)

        conversation_id = uuid.uuid4()
        response = agent.invoke({"input": transcribed_text})
        llm_response = response["output"]

        save_message(db, conversation_id, role="user", content=transcribed_text)
        save_message(db, conversation_id, role="assistant", content=llm_response)
        
        return JSONResponse({
            "transcribed_text": transcribed_text,
            "response": llm_response,
            "conversation_id": str(conversation_id)
        })
    except Exception as e:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        return JSONResponse({"error": f"Ошибка при транскрипции: {str(e)}"}, status_code=500)


@app.post("/tts")
async def text_to_speech(request: Request):
    """Преобразует текст в речь"""
    try:
        data = await request.json()
        text = data.get("text", "")

        if not text:
            return JSONResponse({"error": "Текст не предоставлен"}, status_code=400)


        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )

        return StreamingResponse(
            io.BytesIO(response.content),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        return JSONResponse({
            "error": f"Ошибка при синтезе речи: {str(e)}",
            "use_browser_tts": True
        }, status_code=500)


