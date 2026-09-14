import os,joblib,numpy as np
from config import MODEL_PATH
_model=None
def load_model():
 global _model
 if _model is None and os.path.exists(MODEL_PATH): _model=joblib.load(MODEL_PATH)
 return _model
def predict(features):
 model=load_model()
 if model is not None and hasattr(model,'predict_proba'):
  p=float(model.predict_proba([features])[0][1]); return p,max(p,1-p)
 x=np.asarray(features); variability=float(np.std(x[:20])); spectral=float(np.mean(x[40:80])) if len(x)>=80 else 0
 p=float(np.clip(0.50+0.12*np.tanh(variability/10)-0.08*np.tanh(spectral/20),.05,.95))
 return p,.55
