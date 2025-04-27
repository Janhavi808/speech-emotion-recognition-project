from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf  
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMOTIONS = {
    0: '😊 Happy',
    1: '😢 Sad',
    2: '😠 Angry',
    3: '😐 Neutral',
    4: '😨 Fear',
    5: '🤢 Disgust'
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    """Extract features from audio file for your model"""
    try:    
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        return mfccs_processed
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_emotion(features):
    """Predict emotion from features using your model"""
    probabilities = np.random.rand(6)
    probabilities = probabilities / probabilities.sum()
    
    predicted_index = np.argmax(probabilities)
    predicted_emotion = EMOTIONS.get(predicted_index, '😐 Neutral')
    
    
    confidence_scores = {
        'HAP': float(probabilities[0]) * 100,
        'SAD': float(probabilities[1]) * 100,
        'ANG': float(probabilities[2]) * 100,
        'NEU': float(probabilities[3]) * 100,
        'FEA': float(probabilities[4]) * 100,
        'DIS': float(probabilities[5]) * 100
    }
    
    return predicted_emotion, confidence_scores

@app.route('/')
def index():
    """Serve the main interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio file upload and return emotion prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
          
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if not filename.lower().endswith('.wav'):
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    wav_path = os.path.splitext(file_path)[0] + '.wav'
                    sf.write(wav_path, audio, sr)
                    file_path = wav_path
                except Exception as e:
                    print(f"Could not convert to WAV: {e}")
            
            features = extract_features(file_path)
            if features is None:
                return jsonify({'error': 'Could not process audio file'}), 400
            
            emotion, confidence = predict_emotion(features)
  
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Could not delete temporary file: {e}")
            
            return jsonify({
                'emotion': emotion,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)