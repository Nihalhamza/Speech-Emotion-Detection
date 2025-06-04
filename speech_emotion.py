import numpy as np
import librosa
import joblib
import sounddevice as sd
import scipy.io.wavfile as wav
import os
from tensorflow.keras.models import load_model
import tempfile

# Load model and label encoder
model = load_model(r"C:\Users\nihal\OneDrive\Desktop\DS PROJECTS\DL SINGLE PROJECT\Speech\speech_cnn_model.h5")
label_encoder = joblib.load('label_encoder.pkl')

# Extract MFCC features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Predict emotion
def predict_emotion(audio_path):
    feature = extract_features(audio_path)
    feature = feature.reshape(1, 13, 1, 1)
    prediction = model.predict(feature)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Record from microphone and predict emotion
def detect_emotion_from_microphone(duration=3, fs=22050):
    print(f"\n[Recording] Speak now for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        wav.write(tmpfile.name, fs, recording)
        temp_audio_path = tmpfile.name

    try:
        emotion = predict_emotion(temp_audio_path)
        print(f"[Result] Predicted Emotion: {emotion}\n")
    finally:
        os.remove(temp_audio_path)

if __name__ == "__main__":
    detect_emotion_from_microphone()
