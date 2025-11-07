from sqlalchemy import create_engine, Column, Integer, String, JSON, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker
import uuid
import os
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    skills = Column(JSON)

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autoflush=False, bind=engine, expire_on_commit=False)


