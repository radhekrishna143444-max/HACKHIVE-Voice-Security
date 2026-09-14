# HACKHIVE — Voice Cloning Security MVP

Based on the supplied HACKHIVE architecture: HTML/CSS/JavaScript frontend, FastAPI/Python backend, Librosa/NumPy audio processing, Scikit-learn model, risk scoring, SQLAlchemy + SQLite.

## Run
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
Open http://localhost:8000

## Real model
The API has a clearly marked demo fallback if `model/voice_classifier.pkl` is missing. It is only for demonstrating the application workflow and must not be presented as validated detection accuracy. Train a real classifier with labeled real/synthetic samples using `model/train_model.py`.

## Workflow
Voice → Analysis → AI → Risk → Action
LOW → Allow | MEDIUM → Verify | HIGH → Alert + Verify

The current prototype analyzes uploaded audio. Near-real-time/live-call support should be treated as a deployment/integration layer.
