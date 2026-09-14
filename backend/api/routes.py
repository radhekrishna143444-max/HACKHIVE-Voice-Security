import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from models.database import SessionLocal, VoiceAnalysis
from services.audio_processor import extract_features
from services.ml_model import predict
from services.risk_scorer import assess

router = APIRouter(prefix='/api')
ALLOWED = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}

@router.post('/analyze')
async def analyze_voice(audio_file: UploadFile = File(...)):
    ext = os.path.splitext(audio_file.filename or '')[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, 'Unsupported audio format.')
    data = await audio_file.read()
    if not data:
        raise HTTPException(400, 'Audio file is empty.')
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, 'Audio file is too large. Maximum size is 25 MB.')
    try:
        features, sr = extract_features(data)
        p, c = predict(features)
        risk, reason, reco = assess(p, c)
    except Exception as e:
        raise HTTPException(422, f'Audio analysis failed: {e}')
    db = SessionLocal()
    try:
        row = VoiceAnalysis(filename=audio_file.filename, file_size_bytes=len(data), audio_format=ext[1:], sample_rate=sr,
                            ai_probability=p, ai_confidence=c, risk_level=risk, risk_reason=reason, recommendation=reco)
        db.add(row); db.commit(); db.refresh(row)
        return {'id': row.id, 'filename': row.filename, 'risk_level': risk, 'ai_probability': p,
                'ai_confidence': c, 'risk_reason': reason, 'recommendation': reco}
    finally:
        db.close()

@router.get('/health')
def health():
    return {'status': 'ok', 'service': 'HACKHIVE Voice Security API'}

@router.get('/results')
def results(limit: int = Query(20, ge=1, le=100)):
    db = SessionLocal()
    try:
        rows = db.query(VoiceAnalysis).order_by(VoiceAnalysis.id.desc()).limit(limit).all()
        return [{'id': r.id, 'filename': r.filename, 'risk_level': r.risk_level,
                 'ai_probability': r.ai_probability, 'ai_confidence': r.ai_confidence,
                 'recommendation': r.recommendation, 'created_at': r.analysis_timestamp.isoformat() if r.analysis_timestamp else None} for r in rows]
    finally:
        db.close()

@router.get('/results/{analysis_id}')
def result(analysis_id: int):
    db = SessionLocal()
    try:
        r = db.get(VoiceAnalysis, analysis_id)
        if not r: raise HTTPException(404, 'Analysis not found.')
        return {'id': r.id, 'filename': r.filename, 'risk_level': r.risk_level,
                'ai_probability': r.ai_probability, 'ai_confidence': r.ai_confidence,
                'risk_reason': r.risk_reason, 'recommendation': r.recommendation}
    finally:
        db.close()

@router.post('/verification')
def verification():
    return {'status': 'verification_required', 'message': 'Additional identity verification should be completed before continuing the interaction.'}
