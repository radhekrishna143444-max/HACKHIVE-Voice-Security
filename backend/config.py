import os
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATABASE_URL=f"sqlite:///{os.path.join(BASE_DIR,'hackhive.db')}"
MODEL_PATH=os.path.join(os.path.dirname(BASE_DIR),'model','voice_classifier.pkl')
LOW_THRESHOLD=0.40
HIGH_THRESHOLD=0.70
