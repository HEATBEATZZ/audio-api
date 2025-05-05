# Add this at the top to fix NumPy deprecation issues
import numpy as np
import warnings

# Handle NumPy deprecation and suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Regular imports
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import uuid
import librosa
import soundfile as sf
import threading
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create upload and processed directories
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Track processing progress
processing_status = {}

@app.route('/healthz')
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "Audio processing API is running",
        "models": "available"
    })

def process_audio_task(audio_path, output_path, options, job_id):
    """Background task to process audio"""
    try:
        # Update status to processing
        processing_status[job_id] = {'status': 'processing', 'progress': 0}
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=44100)
        original_audio = y.copy()
        
        # Process based on enabled options
        processing_status[job_id]['progress'] = 10
        
        # Apply audio processing based on options
        if options.get('removeNoise', False):
            processing_status[job_id]['progress'] = 30
            # Noise reduction processing would go here
            pass

        if options.get('removeReverb', False):
            processing_status[job_id]['progress'] = 50
            # Reverb removal processing would go here
            pass

        if options.get('removeInstrumental', False):
            processing_status[job_id]['progress'] = 70
            # Instrumental removal processing would go here
            pass

        if options.get('removeBackgroundVocals', False):
            processing_status[job_id]['progress'] = 90
            # Background vocals removal would go here
            pass
        
        # Save processed audio with explicit format
        sf.write(output_path, y, sr, format='WAV')
        
        # Update status to complete
        processing_status[job_id] = {'status': 'complete', 'progress': 100}
        
    except Exception as e:
        processing_status[job_id] = {'status': 'error', 'error': str(e)}
        print(f"Processing error: {str(e)}")

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get processing options
    options = json.loads(request.form.get('options', '{}'))
    
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
    audio_file.save(audio_path)
    
    # Create output path
    output_filename = f"processed_{job_id}_{filename}"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_audio_task,
        args=(audio_path, output_path, options, job_id)
    )
    thread.daemon = True
    thread.start()
    
    # Return job ID for progress tracking
    return jsonify({
        'job_id': job_id,
        'status': 'processing',
        'audioUrl': f"/download/{job_id}",
        'downloadUrl': f"/download/{job_id}?attachment=true"
    })

@app.route('/process-audio/progress', methods=['GET'])
def check_progress():
    job_id = request.args.get('id')
    if not job_id or job_id not in processing_status:
        return jsonify({'status': 'unknown'}), 404
    
    return jsonify(processing_status[job_id])

@app.route('/download/<job_id>', methods=['GET'])
def download_file(job_id):
    try:
        # Find the file associated with this job ID
        for filename in os.listdir(PROCESSED_FOLDER):
            if filename.startswith(f"processed_{job_id}_"):
                file_path = os.path.join(PROCESSED_FOLDER, filename)
                
                # Determine mime type based on file extension
                mime_type = "audio/mpeg"  # Default to MP3
                if filename.lower().endswith('.wav'):
                    mime_type = "audio/wav"
                elif filename.lower().endswith('.ogg'):
                    mime_type = "audio/ogg"
                elif filename.lower().endswith('.flac'):
                    mime_type = "audio/flac"
                
                # Check if this should be a download or just played in browser
                if request.args.get('attachment'):
                    return send_file(
                        file_path,
                        mimetype=mime_type,
                        as_attachment=True,
                        download_name=filename.replace(f"processed_{job_id}_", "")
                    )
                else:
                    return send_file(
                        file_path,
                        mimetype=mime_type
                    )
        
        return jsonify({'error': 'File not found'}), 404
        
    except Exception as e:
        print(f"Error serving file: {str(e)}")
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
