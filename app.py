from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import uuid
import librosa
import soundfile as sf
import numpy as np
import threading
from werkzeug.utils import secure_filename

# Import your model classes
from models.mdx_net import MDXNetModel
from models.vr_arc import VRArcModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create upload and processed directories
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Track processing progress
processing_status = {}

# Load models
mdx_vocals = MDXNetModel('vocals')
mdx_dereverb = MDXNetModel('dereverb')
mdx_denoiser = MDXNetModel('denoiser')

vr_vocals = VRArcModel('vocals') 
vr_dereverb = VRArcModel('dereverb')
vr_denoiser = VRArcModel('denoiser')

def process_audio_task(audio_path, output_path, options, job_id):
    """Background task to process audio with both MDX-Net and VR-Arc models"""
    try:
        # Update status to processing
        processing_status[job_id] = {'status': 'processing', 'progress': 0}
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=44100)
        original_audio = y.copy()  # Keep a copy of original audio
        
        # Process based on enabled options
        processing_status[job_id]['progress'] = 10
        
        # Process steps based on options
        if options.get('removeNoise'):
            processing_status[job_id]['progress'] = 20
            # Process with both models and blend
            mdx_output = mdx_denoiser.process(y)
            vr_output = vr_denoiser.process(y)
            # Blend results (60% MDX-Net, 40% VR-Arc)
            y = mdx_output * 0.6 + vr_output * 0.4
        
        if options.get('removeReverb'):
            processing_status[job_id]['progress'] = 40
            # Process with both models and blend
            mdx_output = mdx_dereverb.process(y)
            vr_output = vr_dereverb.process(y)
            # Blend results
            y = mdx_output * 0.6 + vr_output * 0.4
        
        if options.get('removeInstrumental'):
            processing_status[job_id]['progress'] = 60
            # Process with both models and blend for vocal isolation
            mdx_vocals_output = mdx_vocals.isolate_vocals(y)
            vr_vocals_output = vr_vocals.isolate_vocals(y)
            # Vocals typically benefit from weighted blending
            y = mdx_vocals_output * 0.7 + vr_vocals_output * 0.3
        
        if options.get('removeBackgroundVocals'):
            processing_status[job_id]['progress'] = 80
            # Similar approach for background vocals
            mdx_bg_removed = mdx_vocals.remove_background_vocals(y)
            vr_bg_removed = vr_vocals.remove_background_vocals(y)
            # Blend results
            y = mdx_bg_removed * 0.5 + vr_bg_removed * 0.5
        
        # Save processed audio
        sf.write(output_path, y, sr)
        
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
    # Find the file associated with this job ID
    for filename in os.listdir(PROCESSED_FOLDER):
        if filename.startswith(f"processed_{job_id}_"):
            file_path = os.path.join(PROCESSED_FOLDER, filename)
            
            # Check if this should be a download or just played in browser
            if request.args.get('attachment'):
                return send_file(file_path, as_attachment=True, download_name=filename.replace(f"processed_{job_id}_", ""))
            else:
                return send_file(file_path)
    
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
