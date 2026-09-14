# Put real samples in training_data/real and synthetic samples in training_data/synthetic.
# Then run this script to create voice_classifier.pkl.
import os,joblib,librosa,numpy as np
from sklearn.ensemble import RandomForestClassifier
BASE=os.path.dirname(__file__)
def f(path):
 y,s=librosa.load(path,sr=16000,mono=True);y=y/(np.max(np.abs(y))+1e-9)
 m=librosa.feature.mfcc(y=y,sr=s,n_mfcc=20);mel=librosa.feature.melspectrogram(y=y,sr=s,n_mels=40);sc=librosa.feature.spectral_centroid(y=y,sr=s);z=librosa.feature.zero_crossing_rate(y)
 return np.concatenate([np.mean(m,1),np.std(m,1),np.mean(librosa.power_to_db(mel),1),[np.mean(sc),np.std(sc),np.mean(z),np.std(z)]])
X=[];Y=[]
for y,label in [('real',0),('synthetic',1)]:
 for name in os.listdir(os.path.join(BASE,'training_data',y)):
  if name.lower().endswith(('.wav','.mp3','.flac','.ogg','.m4a')):X.append(f(os.path.join(BASE,'training_data',y,name)));Y.append(label)
if len(set(Y))<2: raise SystemExit('Add both real and synthetic samples first.')
model=RandomForestClassifier(n_estimators=200,random_state=42,class_weight='balanced');model.fit(X,Y);joblib.dump(model,os.path.join(BASE,'voice_classifier.pkl'));print('Saved voice_classifier.pkl')
