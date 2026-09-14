import io,numpy as np,librosa
def extract_features(data):
 y,sr=librosa.load(io.BytesIO(data),sr=16000,mono=True)
 if len(y)==0: raise ValueError('Audio file is empty.')
 y=y/(np.max(np.abs(y))+1e-9)
 mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20); mel=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=40)
 sc=librosa.feature.spectral_centroid(y=y,sr=sr); z=librosa.feature.zero_crossing_rate(y)
 f=np.concatenate([np.mean(mfcc,1),np.std(mfcc,1),np.mean(librosa.power_to_db(mel),1),[np.mean(sc),np.std(sc),np.mean(z),np.std(z)]])
 return f.astype(np.float32),sr
