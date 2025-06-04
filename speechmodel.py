import librosa
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

dataset_path = r"C:\Users\nihal\OneDrive\Desktop\Zidio\Speech\dataset"
emotions = ['angry', 'fear', 'happy', 'sad', 'neutral', 'disgust', 'surprise']

def load_data(dataset_path, emotions):
    features = []
    labels = []

    for emotion in emotions:
        emotion_folder = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_folder):
            print(f"Warning: Folder for emotion '{emotion}' not found.")
            continue

        for filename in os.listdir(emotion_folder):
            if filename.endswith('.wav'):
                file_path = os.path.join(emotion_folder, filename)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)

    return np.array(features), np.array(labels)

X, y = load_data(dataset_path, emotions)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

X_train_reshaped = X_train.reshape((X_train.shape[0], 13, 1, 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], 13, 1, 1))


model = Sequential()
model.add(Input(shape=(13, 1, 1)))
model.add(Conv2D(32, (1, 1), activation='relu'))
model.add(MaxPooling2D((2, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(emotions), activation='softmax'))

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))

model.save('speech_cnn_model.keras')

test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
print("Test Accuracy: {:.2f}%".format(test_acc * 100))

joblib.dump(label_encoder, 'label_encoder.pkl')
