from sqlalchemy import create_engine,Column,Integer,String,Float,DateTime
from sqlalchemy.orm import declarative_base,sessionmaker
from datetime import datetime
from config import DATABASE_URL
engine=create_engine(DATABASE_URL,connect_args={'check_same_thread':False})
SessionLocal=sessionmaker(bind=engine,autoflush=False,autocommit=False)
Base=declarative_base()
class VoiceAnalysis(Base):
 __tablename__='voice_analysis'
 id=Column(Integer,primary_key=True)
 filename=Column(String); file_size_bytes=Column(Integer); audio_format=Column(String); sample_rate=Column(Integer)
 ai_probability=Column(Float); ai_confidence=Column(Float); risk_level=Column(String); risk_reason=Column(String); recommendation=Column(String)
 analysis_timestamp=Column(DateTime,default=datetime.utcnow)
Base.metadata.create_all(bind=engine)
