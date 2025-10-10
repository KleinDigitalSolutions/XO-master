#!/usr/bin/env python3

import modal
import asyncio
import uuid
import json
import time
import tempfile
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal App Configuration
app = modal.App("audio-enhancement-complete")

# Enhanced Image with dependencies
enhancement_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "fastapi>=0.104.0",
        "python-multipart>=0.0.6",
        "librosa>=0.10.1",
        "soundfile>=0.12.1", 
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "noisereduce>=3.0.0",
        "pyloudnorm>=0.1.1",
        "pedalboard>=0.8.7",
        "requests>=2.31.0",
        "pydub>=0.25.1",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "deepfilternet==0.5.6",
        "openai-whisper>=20231117",
        "tensorflow>=2.13.0",
        "tensorflow-hub>=0.15.0",
    ])
)

# Storage
volume = modal.Volume.from_name("audio-enhancement-volume", create_if_missing=True)
storage_path = "/storage"

# In-memory job status storage
job_status = {}

def save_job_status(job_id: str, status_data: dict):
    """Save job status to both memory and volume"""
    job_status[job_id] = status_data
    
    # Also save to volume for persistence
    status_file = Path(storage_path) / f"status_{job_id}.json"
    try:
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
    except Exception as e:
        print(f"Warning: Could not save status to volume: {e}")

@app.cls(
    image=enhancement_image,
    gpu="a10g",
    timeout=1800,
    memory=8192,
    cpu=4.0,
    volumes={storage_path: volume}
    # Cold start - no warm containers
)
class AudioEnhancer:
    @modal.enter()
    def startup(self):
        """Load all AI models once at container startup"""
        import torch
        import whisper
        import tensorflow_hub as hub
        
        logger.info("🚀 Loading AI models at startup...")
        
        # Load DeepFilterNet (temporarily disabled due to compatibility issues)
        logger.warning("DeepFilterNet temporarily disabled - using enhanced noisereduce instead")
        self.deepfilter_model = None
        self.df_state = None
        
        # Load Whisper
        try:
            logger.info("Loading Whisper...")
            self.whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper loaded successfully")
        except Exception as e:
            logger.warning(f"Whisper loading failed: {e}")
            self.whisper_model = None
        
        # Load YAMNet
        try:
            logger.info("Loading YAMNet...")
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            logger.info("✅ YAMNet loaded successfully")
        except Exception as e:
            logger.warning(f"YAMNet loading failed: {e}")
            self.yamnet_model = None
        
        logger.info("🎉 All AI models loaded successfully!")

    @modal.method()
    def enhance_audio_processing(
        self,
        audio_data: bytes,
        filename: str,
        job_id: str,
        enhancement_options: dict
    ):
        """
        Core audio enhancement processing function
        """
        import io
        import tempfile
        import librosa
        import soundfile as sf
        import numpy as np
        import torch
        import pyloudnorm as pyln
        from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Compressor, NoiseGate, LowShelfFilter, Limiter
        
        try:
            logger.info(f"🎵 Starting audio enhancement job: {job_id}")
            
            # Update status
            save_job_status(job_id, {
                "status": "processing",
                "progress": 10,
                "phase": "Audio wird geladen...",
                "error": None
            })
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Load audio
            audio, sr = librosa.load(temp_path, sr=None, mono=True)
            
            # Ensure appropriate sample rate
            if sr != 48000:
                logger.info(f"🔄 Resampling from {sr}Hz to 48kHz...")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                sr = 48000
            
            enhanced_audio = audio.copy()
            processing_log = []
            
            # Update status
            save_job_status(job_id, {
                "status": "processing",
                "progress": 25,
                "phase": "Rauschunterdrückung wird angewendet...",
                "error": None
            })
            
            # 1. AI-Powered Noise Reduction with DeepFilterNet
            if enhancement_options.get("noise_reduction", True):
                logger.info("🤖 Applying AI-powered noise reduction with DeepFilterNet...")
                try:
                    # DeepFilterNet temporarily disabled - use enhanced noisereduce
                    raise Exception("DeepFilterNet temporarily disabled")
                    
                except Exception as e:
                    logger.warning(f"DeepFilterNet failed, falling back to noisereduce: {e}")
                    try:
                        import noisereduce as nr
                        # 2024 Optimized Parameters for Aggressive Speech Enhancement
                        enhanced_audio = nr.reduce_noise(
                            y=enhanced_audio, 
                            sr=sr, 
                            prop_decrease=1.0,          # Maximum reduction (was 0.9)
                            stationary=False,           # Non-stationary for speech
                            n_std_thresh_stationary=1.0,  # Lower threshold = more aggressive (was 1.5)
                            n_fft=2048,                # Larger FFT for better frequency resolution
                            win_length=2048,           # Matching window length
                            hop_length=512,            # Smaller hop for better time resolution
                            freq_mask_smooth_hz=500,   # Frequency smoothing
                            time_mask_smooth_ms=50     # Time smoothing for speech
                        )
                        processing_log.append("✅ Fallback noise reduction applied")
                    except Exception as e2:
                        logger.warning(f"All noise reduction failed: {e2}")
                        processing_log.append("⚠️ Noise reduction skipped")
            
            # Update status
            save_job_status(job_id, {
                "status": "processing",
                "progress": 50,
                "phase": "Volume wird normalisiert...",
                "error": None
            })
            
            # 2. Volume Normalization
            if enhancement_options.get("volume_normalization", True):
                logger.info("📊 Normalizing volume...")
                try:
                    rms = np.sqrt(np.mean(enhanced_audio**2))
                    if rms > 0:
                        target_rms = 0.2
                        enhanced_audio = enhanced_audio * (target_rms / rms)
                    
                    max_val = np.max(np.abs(enhanced_audio))
                    if max_val > 0.95:
                        enhanced_audio = enhanced_audio * (0.95 / max_val)
                    
                    processing_log.append("✅ Volume normalized")
                except Exception as e:
                    logger.warning(f"Volume normalization failed: {e}")
                    processing_log.append("⚠️ Volume normalization skipped")
            
            # 2.5 AI Filler Word Detection and Removal with Whisper
            if enhancement_options.get("filler_word_removal", True):
                logger.info("🎙️ Detecting and removing filler words with Whisper...")
                try:
                    # Use pre-loaded Whisper model
                    if self.whisper_model is not None:
                        # Create temporary file for Whisper
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_whisper:
                            sf.write(temp_whisper.name, enhanced_audio, sr)
                            
                            # Transcribe with word-level timestamps
                            result = self.whisper_model.transcribe(temp_whisper.name, word_timestamps=True)
                            
                            # Common German and English filler words
                            filler_words = ['äh', 'ähm', 'eh', 'ehm', 'um', 'uh', 'like', 'you know', 'basically', 'actually']
                            
                            # Remove filler words from audio
                            if 'segments' in result:
                                for segment in result['segments']:
                                    if 'words' in segment:
                                        for word in segment['words']:
                                            if word['word'].strip().lower() in filler_words:
                                                start_sample = int(word['start'] * sr)
                                                end_sample = int(word['end'] * sr)
                                                if start_sample < len(enhanced_audio) and end_sample <= len(enhanced_audio):
                                                    # Fade out filler word
                                                    fade_samples = min(int(0.05 * sr), (end_sample - start_sample) // 2)
                                                    enhanced_audio[start_sample:start_sample+fade_samples] *= np.linspace(1, 0, fade_samples)
                                                    enhanced_audio[start_sample+fade_samples:end_sample-fade_samples] = 0
                                                    enhanced_audio[end_sample-fade_samples:end_sample] *= np.linspace(0, 1, fade_samples)
                            
                            os.unlink(temp_whisper.name)
                            processing_log.append("✅ AI filler word removal applied")
                    else:
                        raise Exception("Whisper model not loaded")
                        
                except Exception as e:
                    logger.warning(f"Whisper filler word removal failed: {e}")
                    processing_log.append("⚠️ Filler word removal skipped")
            
            # 2.6 Silence Trimming
            if enhancement_options.get("silence_trimming", True):
                logger.info("✂️ Trimming silence...")
                try:
                    enhanced_audio, _ = librosa.effects.trim(enhanced_audio, top_db=30)
                    processing_log.append("✅ Silence trimmed")
                except Exception as e:
                    logger.warning(f"Silence trimming failed: {e}")
                    processing_log.append("⚠️ Silence trimming skipped")

            # Update status
            save_job_status(job_id, {
                "status": "processing",
                "progress": 75,
                "phase": "Audio-Qualität wird verbessert...",
                "error": None
            })
            
            # 3. AI Cough and Click Detection with YAMNet
            if enhancement_options.get("cough_click_removal", True):
                logger.info("🤧 Detecting and removing coughs/clicks with YAMNet...")
                try:
                    # Use pre-loaded YAMNet model
                    if self.yamnet_model is not None:
                        # Prepare audio for YAMNet (expects 16kHz)
                        yamnet_audio = librosa.resample(enhanced_audio, orig_sr=sr, target_sr=16000)
                        
                        # Get predictions
                        scores, embeddings, spectrogram = self.yamnet_model(yamnet_audio)
                        
                        # YAMNet class indices for unwanted sounds
                        cough_classes = [309, 310]  # Cough, Sneeze
                        click_classes = [379, 380, 381]  # Click, Tick, Tap
                        
                        unwanted_classes = cough_classes + click_classes
                        
                        # Find segments with unwanted sounds
                        for i, score in enumerate(scores):
                            for class_idx in unwanted_classes:
                                if score[class_idx] > 0.3:  # Confidence threshold
                                    # Calculate time segment (YAMNet uses 0.96s windows)
                                    start_time = i * 0.96
                                    end_time = (i + 1) * 0.96
                                    
                                    start_sample = int(start_time * sr)
                                    end_sample = int(end_time * sr)
                                    
                                    if start_sample < len(enhanced_audio) and end_sample <= len(enhanced_audio):
                                        # Apply gentle fade to remove unwanted sound
                                        fade_samples = int(0.1 * sr)  # 100ms fade
                                        enhanced_audio[start_sample:start_sample+fade_samples] *= np.linspace(1, 0.1, fade_samples)
                                        enhanced_audio[start_sample+fade_samples:end_sample-fade_samples] *= 0.1
                                        enhanced_audio[end_sample-fade_samples:end_sample] *= np.linspace(0.1, 1, fade_samples)
                        
                        processing_log.append("✅ AI cough/click removal applied")
                    else:
                        raise Exception("YAMNet model not loaded")
                    
                except Exception as e:
                    logger.warning(f"YAMNet cough/click removal failed: {e}")
                    processing_log.append("⚠️ Cough/click removal skipped")
            
            # 4. Professional Quality Enhancement Chain
            if enhancement_options.get("quality_enhancement", True):
                logger.info("✨ Applying professional audio enhancement...")
                try:
                    # Professional speech enhancement chain based on 2024 best practices
                    board = Pedalboard([
                        # Noise gate to remove low-level noise
                        NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
                        # High-pass filter to remove rumble
                        HighpassFilter(cutoff_frequency_hz=100),
                        # Low shelf boost for clarity
                        LowShelfFilter(cutoff_frequency_hz=400, gain_db=6, q=1),
                        # Professional compressor for broadcast
                        Compressor(threshold_db=-18, ratio=6, attack_ms=3, release_ms=150),
                        # Gentle high-frequency roll-off
                        LowpassFilter(cutoff_frequency_hz=12000),
                        # Final limiter to prevent clipping
                        Limiter(threshold_db=-1.0, release_ms=50)
                    ])
                    enhanced_audio = board(enhanced_audio, sr)
                    processing_log.append("✅ Professional audio enhancement applied")
                except Exception as e:
                    logger.warning(f"Quality enhancement failed: {e}")
                    processing_log.append("⚠️ Quality enhancement skipped")
            

            # Update status
            save_job_status(job_id, {
                "status": "processing",
                "progress": 90,
                "phase": "Loudness wird angepasst...",
                "error": None
            })
            
            # 5. Professional Loudness Targeting (Streaming/Podcast Standard)
            if enhancement_options.get("loudness_targeting", True):
                logger.info("🎚️ Applying streaming/podcast loudness targeting...")
                try:
                    target_lufs = enhancement_options.get("target_lufs", -16.0)  # Streaming/Podcast standard (was -23.0 broadcast)
                    
                    # Create professional meter with optimal block size
                    meter = pyln.Meter(sr, block_size=0.400)  # 400ms blocks per ITU-R BS.1770-4
                    loudness = meter.integrated_loudness(enhanced_audio)
                    
                    if np.isfinite(loudness) and loudness > -70:  # Valid loudness measurement
                        # Apply broadcast-quality loudness normalization
                        enhanced_audio = pyln.normalize.loudness(enhanced_audio, loudness, target_lufs)
                        
                        # Peak normalize to prevent clipping with headroom
                        peak = np.max(np.abs(enhanced_audio))
                        if peak > 0.95:
                            enhanced_audio = enhanced_audio * (0.95 / peak)
                        
                        processing_log.append(f"✅ Streaming/podcast loudness normalized to {target_lufs} LUFS")
                    else:
                        processing_log.append("⚠️ Loudness targeting skipped (insufficient signal)")
                except Exception as e:
                    logger.warning(f"Loudness targeting failed: {e}")
                    processing_log.append("⚠️ Loudness targeting skipped")
            
            # Calculate metrics
            original_rms = np.sqrt(np.mean(audio**2))
            enhanced_rms = np.sqrt(np.mean(enhanced_audio**2))
            
            def estimate_snr(audio_signal):
                sorted_audio = np.sort(np.abs(audio_signal))
                signal_level = np.mean(sorted_audio[-int(len(sorted_audio)*0.1):])
                noise_level = np.mean(sorted_audio[:int(len(sorted_audio)*0.1)])
                return 20 * np.log10(signal_level / max(noise_level, 1e-10))
            
            original_snr = estimate_snr(audio)
            enhanced_snr = estimate_snr(enhanced_audio)
            
            metrics = {
                "original_rms": float(original_rms),
                "enhanced_rms": float(enhanced_rms),
                "rms_improvement_db": float(20 * np.log10(enhanced_rms / max(original_rms, 1e-10))),
                "original_snr_db": float(original_snr),
                "enhanced_snr_db": float(enhanced_snr),
                "snr_improvement_db": float(enhanced_snr - original_snr),
                "dynamic_range_db": float(20 * np.log10(np.max(np.abs(enhanced_audio)) / max(enhanced_rms, 1e-10)))
            }
            
            # Save enhanced audio
            enhanced_filename = f"enhanced_{job_id}.wav"
            enhanced_path = Path(storage_path) / enhanced_filename
            sf.write(enhanced_path, enhanced_audio, sr, subtype='PCM_24')
            
            # Create comparison preview
            min_length = min(len(audio), len(enhanced_audio))
            preview_audio = np.column_stack([audio[:min_length], enhanced_audio[:min_length]])
            preview_filename = f"preview_{job_id}.wav"
            preview_path = Path(storage_path) / preview_filename
            sf.write(preview_path, preview_audio, sr, subtype='PCM_16')
            
            # Final status update
            result = {
                "status": "completed",
                "progress": 100,
                "phase": "Enhancement abgeschlossen!",
                "error": None,
                "job_id": job_id,
                "processing_log": processing_log,
                "metrics": metrics,
                "files": {
                    "enhanced": enhanced_filename,
                    "preview": preview_filename
                },
                "processing_time": None
            }
            
            save_job_status(job_id, result)
            
            # Cleanup
            os.unlink(temp_path)
            
            logger.info(f"✅ Audio enhancement completed for job: {job_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in audio enhancement: {str(e)}")
            error_result = {
                "status": "error",
                "progress": 0,
                "phase": "Fehler aufgetreten",
                "error": str(e),
                "job_id": job_id
            }
            save_job_status(job_id, error_result)
            return error_result

# FastAPI Web App
web_app = FastAPI(title="Audio Enhancement API")

# Add CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create class instance
enhancer = AudioEnhancer()

@app.function(
    image=enhancement_image,
    volumes={storage_path: volume}
)
@modal.asgi_app()
def fastapi_app():
    
    @web_app.post("/upload/enhancement")
    async def upload_audio_enhancement(
        audio: UploadFile = File(...),
        enhancement_options: str = Form(...)
    ):
        """Upload audio for enhancement processing"""
        try:
            # Validate file format
            allowed_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']
            if not any(audio.filename.lower().endswith(ext) for ext in allowed_extensions):
                raise HTTPException(400, f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}")
            
            # Read audio data
            audio_data = await audio.read()
            
            if len(audio_data) == 0:
                raise HTTPException(400, "Empty audio file uploaded")
            
            # File size limit (100MB)
            max_size = 100 * 1024 * 1024
            if len(audio_data) > max_size:
                raise HTTPException(400, f"File too large (max {max_size // 1024 // 1024}MB)")
            
            # Parse enhancement options
            try:
                options = json.loads(enhancement_options)
            except json.JSONDecodeError:
                raise HTTPException(400, "Invalid enhancement options JSON")
            
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            print(f"Starting enhancement for {audio.filename} ({len(audio_data)} bytes)")
            
            # Initialize job status
            save_job_status(job_id, {
                "status": "queued",
                "progress": 0,
                "phase": "Upload abgeschlossen, Enhancement wird vorbereitet...",
                "error": None,
                "job_id": job_id
            })
            
            # Start async processing using class method
            enhancer.enhance_audio_processing.spawn(audio_data, audio.filename, job_id, options)
            
            # Return job ID for status tracking
            response = {
                "success": True,
                "job_id": job_id,
                "message": "Enhancement started successfully"
            }
            
            print(f"Enhancement job {job_id} started successfully")
            return JSONResponse(content=response, status_code=200)
                
        except Exception as e:
            print(f"Enhancement upload error: {str(e)}")
            raise HTTPException(500, f"Enhancement upload error: {str(e)}")

    @web_app.get("/status/enhancement/{job_id}")
    async def enhancement_status(job_id: str):
        """Status API for enhancement progress"""
        # Always check volume first for latest status
        volume.reload()
        status_file = Path(storage_path) / f"status_{job_id}.json"
        
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)
                print(f"Enhancement status loaded from volume for job {job_id}: {status.get('progress', 0)}% - {status.get('phase', 'Unknown')}")
                return status
            except Exception as e:
                print(f"Error reading status from volume: {e}")
        
        # Fallback to memory
        if job_id in job_status:
            status = job_status[job_id]
            print(f"Enhancement status from memory for job {job_id}: {status.get('progress', 0)}% - {status.get('phase', 'Unknown')}")
            return status
        
        print(f"Enhancement job {job_id} not found")
        raise HTTPException(404, f"Job {job_id} not found")

    @web_app.get("/download/enhancement/{job_id}/{filename}")
    async def download_enhancement_file(job_id: str, filename: str):
        """Download enhanced file"""
        volume.reload()
        file_path = Path(storage_path) / filename
        
        if not file_path.exists():
            raise HTTPException(404, f"File {filename} not found for job {job_id}")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="audio/wav"
        )

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "audio-enhancement"}
    
    return web_app

if __name__ == "__main__":
    print("Klein Digital Solutions - Audio Enhancement Service")
    print("Ready for deployment!")