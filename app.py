from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from pydub import AudioSegment  # Library for converting audio formats

app = Flask(__name__)
CORS(app)  # Apply CORS to allow requests from all origins

# Load your trained model
MODEL_PATH = "audio_classifier.h5"
model = load_model(MODEL_PATH)

# Define parameters for audio processing
SAMPLE_RATE = 16000  # Sample rate of audio files
DURATION = 5  # Duration of audio clips in seconds
N_MELS = 128  # Number of Mel frequency bins
MAX_TIME_STEPS = 109  # Same max time steps used during training

def preprocess_audio(file_path):
    """
    Preprocess the audio file to create a Mel spectrogram with the same shape
    expected by the model.
    """
    # Load audio file using librosa
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    
    # Extract Mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    # Add channel dimension
    mel_spectrogram = mel_spectrogram[..., np.newaxis]
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension

    return mel_spectrogram

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict whether an audio file is real or fake.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    # Get the uploaded file from the request
    file = request.files['file']
    file_extension = os.path.splitext(file.filename)[1].lower()

    file_path = './temp_audio.flac'  # Default to flac format

    # Check if the uploaded file is in mp3 format and convert it to flac
    if file_extension == '.mp3':
        # Save the mp3 file temporarily
        temp_mp3_path = './temp_audio.mp3'
        file.save(temp_mp3_path)

        # Convert mp3 to flac using pydub
        audio = AudioSegment.from_mp3(temp_mp3_path)
        audio.export(file_path, format='flac')

        # Remove the temporary mp3 file
        os.remove(temp_mp3_path)

    elif file_extension == '.flac':
        # If the file is already flac, save it directly
        file.save(file_path)

    else:
        return jsonify({'error': 'Unsupported file format. Please upload a .flac or .mp3 file.'}), 400

    # Preprocess the audio file
    mel_spectrogram = preprocess_audio(file_path)
    
    # Make a prediction using the pre-trained model
    prediction = model.predict(mel_spectrogram)
    class_labels = ['Fake', 'Real']
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Remove the temporary flac file after prediction
    os.remove(file_path)

    # Return the result as JSON
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)
