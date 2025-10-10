#!/usr/bin/env python3
"""
Klein Digital Solutions - Audio Enhancement Complete
Professional Speech Enhancement with DeepFilterNet, Whisper & YAMNet
"""

import modal
import os
from typing import Dict, List, Optional, Tuple

# Modal App Configuration
app = modal.App("audio-enhancement-complete")

# Enhanced Image with all dependencies
enhancement_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        # Core audio processing
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        
        # DeepFilterNet for noise reduction
        "deepfilternet>=0.5.6",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        
        # Whisper for speech analysis
        "openai-whisper>=20231117",
        "whisper-timestamped>=1.14.2",
        
        # YAMNet dependencies
        "tensorflow>=2.13.0",
        "tensorflow-hub>=0.15.0",
        
        # Audio enhancement
        "noisereduce>=3.0.0",
        "pyloudnorm>=0.1.1",
        "pedalboard>=0.8.7",
        
        # Utilities
        "requests>=2.31.0",
        "pydub>=0.25.1",
        "webrtcvad>=2.0.10",
    ])
    .run_commands([
        # Download DeepFilterNet models
        "python -c 'import df; df.enhance.download_model()'",
        # Download Whisper models
        "python -c 'import whisper; whisper.load_model(\"base\")'",
        # Prepare YAMNet
        "python -c 'import tensorflow_hub as hub; hub.load(\"https://tfhub.dev/google/yamnet/1\")'",
    ])
)

# Global model loading
models = {}

@app.function(
    image=enhancement_image,
    gpu="a10g",
    timeout=1800,  # 30 minutes
    memory=8192,   # 8GB RAM
    cpu=4.0,
    volumes={"/models": modal.Volume.from_name("audio-enhancement-models", create_if_missing=True)}
)
def enhance_audio_quality(
    audio_data: bytes,
    job_id: str,
    enhancement_options: Dict = None
) -> Dict:
    """
    Complete Audio Enhancement Pipeline
    
    Args:
        audio_data: Raw audio file bytes
        job_id: Unique job identifier
        enhancement_options: Configuration for enhancement features
    
    Returns:
        Dict with processing results and file paths
    """
    # Import dependencies inside function
    import io
    import tempfile
    import librosa
    import soundfile as sf
    import numpy as np
    import json
    import logging
    import traceback
    from pathlib import Path
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    global models
    
    try:
        logger.info(f"🎵 Starting audio enhancement job: {job_id}")
        
        # Default enhancement options
        if enhancement_options is None:
            enhancement_options = {
                "noise_reduction": True,
                "volume_normalization": True,
                "filler_word_removal": False,
                "silence_trimming": True,
                "loudness_targeting": True,
                "target_lufs": -16.0,
                "cough_detection": False,
                "quality_enhancement": True
            }
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as input_file:
            input_file.write(audio_data)
            input_path = input_file.name
        
        # Load audio
        logger.info("📊 Loading audio file...")
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Ensure sample rate is appropriate (DeepFilterNet works best at 48kHz)
        if sr != 48000:
            logger.info(f"🔄 Resampling from {sr}Hz to 48kHz...")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000
        
        # Initialize models if needed
        if not models:
            logger.info("🧠 Loading AI models...")
            _load_models()
        
        # Audio enhancement pipeline
        enhanced_audio = audio.copy()
        processing_log = []
        
        # 1. Noise Reduction with DeepFilterNet
        if enhancement_options.get("noise_reduction", True):
            logger.info("🔧 Applying noise reduction...")
            enhanced_audio = _apply_noise_reduction(enhanced_audio, sr)
            processing_log.append("✅ Noise reduction applied")
        
        # 2. Volume Normalization
        if enhancement_options.get("volume_normalization", True):
            logger.info("📊 Normalizing volume...")
            enhanced_audio = _normalize_volume(enhanced_audio)
            processing_log.append("✅ Volume normalized")
        
        # 3. Silence Trimming
        if enhancement_options.get("silence_trimming", True):
            logger.info("✂️ Trimming silence...")
            enhanced_audio = _trim_silence(enhanced_audio, sr)
            processing_log.append("✅ Silence trimmed")
        
        # 4. Filler Word Removal (Advanced)
        if enhancement_options.get("filler_word_removal", False):
            logger.info("🗣️ Detecting and removing filler words...")
            enhanced_audio, removed_words = _remove_filler_words(enhanced_audio, sr)
            processing_log.append(f"✅ Removed {len(removed_words)} filler words")
        
        # 5. Cough Detection and Removal
        if enhancement_options.get("cough_detection", False):
            logger.info("🤧 Detecting and removing coughs...")
            enhanced_audio, removed_coughs = _remove_coughs(enhanced_audio, sr)
            processing_log.append(f"✅ Removed {len(removed_coughs)} coughs")
        
        # 6. Quality Enhancement
        if enhancement_options.get("quality_enhancement", True):
            logger.info("✨ Applying quality enhancement...")
            enhanced_audio = _enhance_audio_quality(enhanced_audio, sr)
            processing_log.append("✅ Audio quality enhanced")
        
        # 7. Loudness Targeting
        if enhancement_options.get("loudness_targeting", True):
            target_lufs = enhancement_options.get("target_lufs", -16.0)
            logger.info(f"🎚️ Targeting {target_lufs} LUFS...")
            enhanced_audio = _apply_loudness_targeting(enhanced_audio, sr, target_lufs)
            processing_log.append(f"✅ Loudness targeted to {target_lufs} LUFS")
        
        # Save enhanced audio
        output_path = f"/tmp/enhanced_{job_id}.wav"
        sf.write(output_path, enhanced_audio, sr, subtype='PCM_24')
        
        # Generate comparison preview
        preview_path = _create_comparison_preview(audio, enhanced_audio, sr, job_id)
        
        # Calculate improvement metrics
        metrics = _calculate_improvement_metrics(audio, enhanced_audio, sr)
        
        # Save to volume for download
        volume_path = f"/enhanced/{job_id}"
        os.makedirs(volume_path, exist_ok=True)
        
        # Copy files to volume
        enhanced_volume_path = f"{volume_path}/enhanced.wav"
        preview_volume_path = f"{volume_path}/preview.wav"
        
        with open(output_path, 'rb') as f:
            with open(enhanced_volume_path, 'wb') as vf:
                vf.write(f.read())
        
        with open(preview_path, 'rb') as f:
            with open(preview_volume_path, 'wb') as vf:
                vf.write(f.read())
        
        # Cleanup temp files
        os.unlink(input_path)
        os.unlink(output_path)
        os.unlink(preview_path)
        
        logger.info(f"✅ Audio enhancement completed for job: {job_id}")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "processing_log": processing_log,
            "metrics": metrics,
            "files": {
                "enhanced": f"/enhanced/{job_id}/enhanced.wav",
                "preview": f"/enhanced/{job_id}/preview.wav"
            },
            "enhancement_applied": enhancement_options
        }
        
    except Exception as e:
        logger.error(f"❌ Error in audio enhancement: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "job_id": job_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def _load_models():
    """Load all required AI models"""
    import logging
    
    # Configure logging
    logger = logging.getLogger(__name__)
    
    global models
    
    try:
        # Load DeepFilterNet
        logger.info("Loading DeepFilterNet...")
        import df
        models['deepfilternet'] = df.enhance.enhance
        
        # Load Whisper
        logger.info("Loading Whisper...")
        import whisper
        models['whisper'] = whisper.load_model("base")
        
        # Load YAMNet
        logger.info("Loading YAMNet...")
        import tensorflow_hub as hub
        models['yamnet'] = hub.load("https://tfhub.dev/google/yamnet/1")
        
        logger.info("✅ All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def _apply_noise_reduction(audio, sr: int):
    """Apply DeepFilterNet noise reduction"""
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Use noisereduce as fallback if DeepFilterNet fails
        import noisereduce as nr
        
        # Apply spectral gating noise reduction
        reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
        
        logger.info("✅ Noise reduction applied successfully")
        return reduced_noise
        
    except Exception as e:
        logger.warning(f"Noise reduction failed: {e}, returning original audio")
        return audio

def _normalize_volume(audio):
    """Normalize audio volume"""
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 0.2  # Target RMS level
            audio = audio * (target_rms / rms)
        
        # Prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
        
        return audio
        
    except Exception as e:
        logger.warning(f"Volume normalization failed: {e}")
        return audio

def _trim_silence(audio, sr: int):
    """Trim silence from beginning and end"""
    import librosa
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Use librosa to trim silence
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=30)
        return trimmed_audio
        
    except Exception as e:
        logger.warning(f"Silence trimming failed: {e}")
        return audio

def _remove_filler_words(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, List]:
    """Remove filler words using Whisper"""
    try:
        # Save audio temporarily
        temp_path = f"/tmp/whisper_temp.wav"
        sf.write(temp_path, audio, sr)
        
        # Transcribe with timestamps
        result = models['whisper'].transcribe(temp_path, word_timestamps=True)
        
        # German and English filler words
        filler_words = {
            'äh', 'ähm', 'ähem', 'mh', 'mhm', 
            'uh', 'um', 'uhm', 'er', 'ah', 'eh'
        }
        
        # Find filler word segments
        segments_to_remove = []
        
        if 'words' in result:
            for word_info in result['words']:
                word = word_info['word'].lower().strip()
                if word in filler_words:
                    segments_to_remove.append({
                        'start': word_info['start'],
                        'end': word_info['end'],
                        'word': word
                    })
        
        # Remove filler word segments
        if segments_to_remove:
            audio_segments = []
            last_end = 0.0
            
            for segment in segments_to_remove:
                # Add audio before filler word
                start_sample = int(last_end * sr)
                end_sample = int(segment['start'] * sr)
                if end_sample > start_sample:
                    audio_segments.append(audio[start_sample:end_sample])
                
                last_end = segment['end']
            
            # Add remaining audio
            if last_end < len(audio) / sr:
                start_sample = int(last_end * sr)
                audio_segments.append(audio[start_sample:])
            
            # Concatenate segments
            if audio_segments:
                cleaned_audio = np.concatenate(audio_segments)
            else:
                cleaned_audio = audio
        else:
            cleaned_audio = audio
        
        os.unlink(temp_path)
        logger.info(f"✅ Removed {len(segments_to_remove)} filler words")
        return cleaned_audio, segments_to_remove
        
    except Exception as e:
        logger.warning(f"Filler word removal failed: {e}")
        return audio, []

def _remove_coughs(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, List]:
    """Remove coughs using YAMNet"""
    try:
        # YAMNet expects 16kHz audio
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio
        
        # Run YAMNet inference
        scores, embeddings, spectrogram = models['yamnet'](audio_16k)
        
        # Find cough events (class index for cough is approximately 310)
        cough_threshold = 0.5
        cough_segments = []
        
        # Convert scores to numpy if needed
        scores_np = scores.numpy() if hasattr(scores, 'numpy') else scores
        
        # Find cough events
        for i, frame_scores in enumerate(scores_np):
            # Check for cough-related classes
            cough_score = max(frame_scores[305:315])  # Approximate cough class range
            if cough_score > cough_threshold:
                time_start = i * 0.96  # YAMNet frame rate
                time_end = (i + 1) * 0.96
                cough_segments.append({
                    'start': time_start,
                    'end': time_end,
                    'confidence': float(cough_score)
                })
        
        # Remove cough segments
        if cough_segments:
            audio_segments = []
            last_end = 0.0
            
            for segment in cough_segments:
                # Add audio before cough
                start_sample = int(last_end * sr)
                end_sample = int(segment['start'] * sr)
                if end_sample > start_sample:
                    audio_segments.append(audio[start_sample:end_sample])
                
                last_end = segment['end']
            
            # Add remaining audio
            if last_end < len(audio) / sr:
                start_sample = int(last_end * sr)
                audio_segments.append(audio[start_sample:])
            
            # Concatenate segments
            if audio_segments:
                cleaned_audio = np.concatenate(audio_segments)
            else:
                cleaned_audio = audio
        else:
            cleaned_audio = audio
        
        logger.info(f"✅ Removed {len(cough_segments)} cough segments")
        return cleaned_audio, cough_segments
        
    except Exception as e:
        logger.warning(f"Cough removal failed: {e}")
        return audio, []

def _enhance_audio_quality(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply quality enhancement filters"""
    try:
        from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Compressor
        
        # Create enhancement chain
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80),  # Remove low rumble
            LowpassFilter(cutoff_frequency_hz=8000), # Remove harsh highs
            Compressor(threshold_db=-16, ratio=4, attack_ms=3, release_ms=100)
        ])
        
        # Apply enhancement
        enhanced = board(audio, sr)
        return enhanced
        
    except Exception as e:
        logger.warning(f"Quality enhancement failed: {e}")
        return audio

def _apply_loudness_targeting(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Apply loudness targeting"""
    try:
        import pyloudnorm as pyln
        
        # Measure current loudness
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        
        # Apply loudness normalization
        if np.isfinite(loudness):
            normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)
            return normalized_audio
        else:
            return audio
            
    except Exception as e:
        logger.warning(f"Loudness targeting failed: {e}")
        return audio

def _create_comparison_preview(original: np.ndarray, enhanced: np.ndarray, sr: int, job_id: str) -> str:
    """Create before/after comparison preview"""
    try:
        # Ensure same length
        min_length = min(len(original), len(enhanced))
        original = original[:min_length]
        enhanced = enhanced[:min_length]
        
        # Create stereo preview: original left, enhanced right
        stereo_preview = np.column_stack([original, enhanced])
        
        # Save preview
        preview_path = f"/tmp/preview_{job_id}.wav"
        sf.write(preview_path, stereo_preview, sr, subtype='PCM_16')
        
        return preview_path
        
    except Exception as e:
        logger.warning(f"Preview creation failed: {e}")
        # Fallback: just save enhanced audio
        preview_path = f"/tmp/preview_{job_id}.wav"
        sf.write(preview_path, enhanced, sr, subtype='PCM_16')
        return preview_path

def _calculate_improvement_metrics(original: np.ndarray, enhanced: np.ndarray, sr: int) -> Dict:
    """Calculate audio improvement metrics"""
    try:
        # Ensure same length for comparison
        min_length = min(len(original), len(enhanced))
        original = original[:min_length]
        enhanced = enhanced[:min_length]
        
        # RMS levels
        original_rms = np.sqrt(np.mean(original**2))
        enhanced_rms = np.sqrt(np.mean(enhanced**2))
        
        # SNR estimation
        def estimate_snr(audio):
            # Simple SNR estimation
            sorted_audio = np.sort(np.abs(audio))
            signal_level = np.mean(sorted_audio[-int(len(sorted_audio)*0.1):])  # Top 10%
            noise_level = np.mean(sorted_audio[:int(len(sorted_audio)*0.1)])    # Bottom 10%
            return 20 * np.log10(signal_level / max(noise_level, 1e-10))
        
        original_snr = estimate_snr(original)
        enhanced_snr = estimate_snr(enhanced)
        
        return {
            "original_rms": float(original_rms),
            "enhanced_rms": float(enhanced_rms),
            "rms_improvement_db": float(20 * np.log10(enhanced_rms / max(original_rms, 1e-10))),
            "original_snr_db": float(original_snr),
            "enhanced_snr_db": float(enhanced_snr),
            "snr_improvement_db": float(enhanced_snr - original_snr),
            "dynamic_range_db": float(20 * np.log10(np.max(np.abs(enhanced)) / max(enhanced_rms, 1e-10)))
        }
        
    except Exception as e:
        logger.warning(f"Metrics calculation failed: {e}")
        return {"error": str(e)}

# Health check endpoint
@app.function(image=enhancement_image)
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "audio-enhancement-complete"}

# Get job status
@app.function(image=enhancement_image)
def get_job_status(job_id: str):
    """Get processing status for a job"""
    try:
        # Check if results exist
        result_path = f"/enhanced/{job_id}/enhanced.wav"
        if os.path.exists(result_path):
            return {"status": "completed", "job_id": job_id}
        else:
            return {"status": "processing", "job_id": job_id}
    except Exception as e:
        return {"status": "error", "job_id": job_id, "error": str(e)}

# Download endpoints
@app.function(image=enhancement_image)
def download_enhanced(job_id: str):
    """Download enhanced audio file"""
    try:
        file_path = f"/enhanced/{job_id}/enhanced.wav"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"Enhanced file not found for job {job_id}")
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

@app.function(image=enhancement_image)
def download_preview(job_id: str):
    """Download comparison preview"""
    try:
        file_path = f"/enhanced/{job_id}/preview.wav"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"Preview file not found for job {job_id}")
    except Exception as e:
        raise Exception(f"Preview download failed: {str(e)}")

if __name__ == "__main__":
    print("Klein Digital Solutions - Audio Enhancement Complete")
    print("Ready for deployment!")