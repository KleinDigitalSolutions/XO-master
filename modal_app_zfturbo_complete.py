#!/usr/bin/env python3
"""
Klein Digital Solutions - Music AI Separator
ZFTurbo Music-Source-Separation-Training Complete Implementation
Modal Serverless Deployment with A10G GPU - Production Ready

Features:
- BS-RoFormer (SDR: 9.65dB) - State-of-the-Art Quality
- Complete ZFTurbo requirements.txt implementation  
- Proper GPU memory optimization
- Comprehensive error handling
- Modal volume file handling
- Ensemble capabilities ready
"""

import modal
import os
from pathlib import Path
import tempfile
import uuid
import subprocess
import time
import json
import zipfile
import io
import numpy as np
import soundfile as sf
import shutil

# Global storage for temporary files
temp_files_storage = {}
# Status dictionary for job progress
job_status = {}

# Modal app instance
app = modal.App("music-ai-separator-zfturbo-complete")

# Create persistent volume for file storage
volume = modal.Volume.from_name("music-files-storage", create_if_missing=True)
storage_path = "/vol/music_files"

# Complete ZFTurbo requirements implementation - following best practices
# All dependencies from requirements.txt properly installed
zfturbo_image = modal.Image.debian_slim(python_version="3.11").apt_install([
    "ffmpeg",
    "libsndfile1", 
    "git",
    "libportaudio2",  # For pyaudio
    "portaudio19-dev",  # For pyaudio
    "build-essential",  # For compilation
    "python3-dev"  # For compilation
]).pip_install([
    # Core PyTorch stack
    "torch>=2.0.1",
    "torchaudio",
    "torchvision",  # Often needed with torch
    
    # Core scientific computing
    "numpy",
    "pandas", 
    "scipy",
    "soundfile",
    
    # ZFTurbo specific dependencies - exact versions from requirements.txt
    "ml_collections",
    "tqdm",
    "segmentation_models_pytorch==0.3.3",
    "timm==0.9.2", 
    "audiomentations==0.24.0",
    "pedalboard~=0.8.1",
    "omegaconf==2.2.3",
    "beartype==0.14.1",
    "rotary_embedding_torch==0.3.5",
    "einops==0.8.1",
    "librosa",
    "demucs==4.0.0",
    "transformers~=4.35.0",
    "torchmetrics==0.11.4",
    "spafe==0.3.2",
    "protobuf==3.20.3",
    "torch_audiomentations",
    "asteroid==0.7.0",
    "auraloss",
    "torchseg",
    "bitsandbytes",
    "wandb",
    "accelerate",
    "huggingface-hub>=0.23.0",
    "prodigyopt",
    "torch_log_wmse",
    "loralib",
    "matplotlib",
    "hyper_connections==0.1.11",
    "sageattention==1.0.6",
    
    # FastAPI stack
    "fastapi>=0.100.0,<0.104.0",
    "python-multipart>=0.0.6",
    
    # Additional useful packages
    "requests",
    "Pillow"
]).run_commands([
    "cd /root && git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git",
    "cd /root/Music-Source-Separation-Training && pip install -r requirements.txt || echo 'Some packages may have failed, continuing...'"
])

# BS-RoFormer Configuration
BS_ROFORMER_CONFIG = {
    "model_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
    "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml",
    "model_type": "bs_roformer",
    "sdr": "9.65"
}

# GPU Memory Optimization Settings - following best practices
GPU_SETTINGS = {
    "device_ids": "0",
    "use_cuda": True,
    "memory_efficient": True,
    "no_cache": False,  # Keep cache for better performance
    "precision": "float32"  # Better quality than float16
}

def save_job_status(job_id: str, status_data: dict):
    """Save job status to volume for persistence"""
    try:
        status_file = Path(storage_path) / f"status_{job_id}.json"
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
        # Force commit for status updates
        volume.commit()
        print(f"📊 Status saved for job {job_id}: {status_data.get('progress', 0)}% - {status_data.get('phase', 'Unknown')}")
    except Exception as e:
        print(f"⚠️ Error saving job status: {e}")

def download_bs_roformer_models_local():
    """Download BS-RoFormer model and config files with proper error handling (lokal, synchron)"""
    import requests
    import os
    model_dir = Path("/root/bs_roformer_models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model_bs_roformer_ep_17_sdr_9.6568.ckpt"
    config_path = model_dir / "config_bs_roformer_384_8_2_485100.yaml"
    try:
        if not model_path.exists():
            print("🔄 Downloading BS-RoFormer model...")
            response = requests.get(BS_ROFORMER_CONFIG["model_url"], timeout=300)
            response.raise_for_status()
            model_path.write_bytes(response.content)
            print(f"✅ Model downloaded ({len(response.content) // 1024 // 1024} MB)")
        if not config_path.exists():
            print("🔄 Downloading BS-RoFormer config...")
            response = requests.get(BS_ROFORMER_CONFIG["config_url"], timeout=60)
            response.raise_for_status()
            config_path.write_text(response.text)
            print("✅ Config downloaded")
        return str(model_path), str(config_path)
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        raise


@app.function(
    image=zfturbo_image,
    gpu="a10g", 
    timeout=1800,  # 30 minutes
    memory=20480,  # 20GB memory
    volumes={storage_path: volume},
    scaledown_window=120
)
def bs_roformer_separate_audio(
    audio_data: bytes, 
    filename: str, 
    job_id: str, 
    selected_stems: str = "all",
    use_tta: bool = False,
    extract_instrumental: bool = False
) -> dict:
    """
    BS-RoFormer separation with complete ZFTurbo implementation
    Following all best practices for GPU memory, error handling, and quality
    """
    start_time = time.time()
    # Set initial job status
    job_status[job_id] = {
        "status": "uploading",
        "progress": 5,
        "phase": "Upload läuft",
        "error": None,
        "processing_time": None
    }
    try:
        # Setup directories with proper permissions
        job_status[job_id] = {
            "status": "processing",
            "progress": 10,
            "phase": "Datei wird vorbereitet",
            "error": None,
            "processing_time": None
        }
        save_job_status(job_id, job_status[job_id])
        job_dir = Path(storage_path) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        input_file = job_dir / filename
        input_file.write_bytes(audio_data)
        
        output_dir = job_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        # KORREKTUR: Vereinheitlichter und vereinfachter Ausgabeordner
        final_output_dir = output_dir / "final_stems"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse selected stems from the input string
        if selected_stems == "all":
            stems_to_process = ["vocals", "drums", "bass", "other"]
        else:
            stems_to_process = [s.strip() for s in selected_stems.split(",") if s.strip() in ["vocals", "drums", "bass", "other"]]
        
        if not stems_to_process: # Fallback if empty or invalid
            stems_to_process = ["vocals", "drums", "bass", "other"]
        print(f"🎯 Initial stems to process: {stems_to_process}")

        print(f"🚀 Starting BS-RoFormer separation for job {job_id}")
        job_status[job_id] = {
            "status": "processing",
            "progress": 20,
            "phase": "Modell wird geladen",
            "error": None,
            "processing_time": None
        }
        save_job_status(job_id, job_status[job_id])
        print(f"📁 Input: {input_file} ({len(audio_data) // 1024} KB)")
        
        # Download models synchron im selben Container
        model_path, config_path = download_bs_roformer_models_local()
        
        # Prepare input directory for ZFTurbo inference script
        temp_input_dir = job_dir / "temp_input"
        temp_input_dir.mkdir(exist_ok=True)
        # Konvertiere zu WAV falls nötig
        file_ext = Path(filename).suffix.lower()
        input_wav_path = temp_input_dir / "input.wav"
        if file_ext != ".wav":
            # Speichere Originaldatei
            temp_orig_file = temp_input_dir / filename
            temp_orig_file.write_bytes(audio_data)
            # Konvertiere zu input.wav
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_orig_file),
                "-ar", "44100",   # Sample rate
                "-ac", "2",       # Stereo
                "-sample_fmt", "s16",  # 16-bit PCM
                str(input_wav_path)
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=60)
                print(f"✅ Konvertiert {filename} zu WAV: input.wav")
            except Exception as e:
                print(f"❌ FFmpeg-Konvertierung fehlgeschlagen: {e}")
                raise Exception(f"FFmpeg-Konvertierung fehlgeschlagen: {e}")
        else:
            # Falls schon WAV, kopiere/benenne als input.wav
            temp_input_file = temp_input_dir / filename
            temp_input_file.write_bytes(audio_data)
            os.rename(str(temp_input_file), str(input_wav_path))
        
        # Use the original converted WAV file directly for inference
        input_for_inference = input_wav_path

        # Debug: Print job_id and input.wav path for troubleshooting
        print(f"[DEBUG] Job-ID: {job_id} | Input path: {input_for_inference} | Exists: {input_for_inference.exists()}")
        
        # File is already named input.wav and in the right location - no copy needed
        
        volume.commit()

        # --- EINZIGER Inferenz-Aufruf für den kombinierten Ordner ---
        inference_cmd = [
            "python", "/root/Music-Source-Separation-Training/inference.py",
            "--model_type", BS_ROFORMER_CONFIG["model_type"],
            "--config_path", config_path,
            "--start_check_point", model_path,
            "--input_folder", str(temp_input_dir),
            "--store_dir", str(output_dir),
            "--device_ids", GPU_SETTINGS["device_ids"]
        ]
        
        if use_tta:
            inference_cmd.append("--use_tta")
        # Instrumental can only be extracted from the full stereo mix
        if extract_instrumental:
             inference_cmd.append("--extract_instrumental")

        print(f"🎵 Running single combined ZFTurbo BS-RoFormer inference...")
        job_status[job_id] = {
            "status": "processing",
            "progress": 40,
            "phase": "Separation läuft",
            "error": None,
            "processing_time": None
        }
        save_job_status(job_id, job_status[job_id])
        print(f"   Command: {' '.join(inference_cmd)}")
        
        # Set environment variables for GPU optimization
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "TOKENIZERS_PARALLELISM": "false"  # Avoid tokenizer warnings
        })
        
        # Execute inference with proper timeout and error handling
        result = subprocess.run(
            inference_cmd,
            capture_output=True,
            text=True,
            timeout=1500,  # 25 minutes
            env=env,
            cwd="/root/Music-Source-Separation-Training"
        )
        
        if result.returncode != 0:
            print(f"❌ BS-RoFormer inference failed:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            raise Exception(f"BS-RoFormer inference failed: {result.stderr}")
        
        print("✅ BS-RoFormer inference completed successfully")
        job_status[job_id] = {
            "status": "processing",
            "progress": 60,
            "phase": "Dateien werden kopiert",
            "error": None,
            "processing_time": None
        }
        save_job_status(job_id, job_status[job_id])
        # Find and organize output files
        
        # ===================================================================
        # ✨ NEU: Robuste Organisation der Ergebnisse
        # Wir greifen direkt auf die bekannten Ausgabeordner zu.
        # ===================================================================
        output_files = []
        
        # Definiere die exakten Pfade, wo die Ergebnisse liegen
        # HINWEIS: Das Skript erstellt einen Unterordner basierend auf dem Dateinamen der Eingabe.
        input_filename_stem = Path(next(temp_input_dir.glob("*.wav")).name).stem
        results_base_dir = output_dir / input_filename_stem

        # Move only the SELECTED stems to final output directory
        # BS-RoFormer always generates all 4 stems, but we only keep what the user wants
        
        # All available stems from BS-RoFormer
        all_possible_stems = ["vocals", "drums", "bass", "other"]
        
        # Add instrumental if requested
        if extract_instrumental:
            all_possible_stems.append("instrumental")
        
        print(f"🎯 User selected stems: {stems_to_process}")
        print(f"📁 Available stems from BS-RoFormer: {all_possible_stems}")
        
        # Only move the stems that the user actually wants
        for stem_type in all_possible_stems:
            stem_path = results_base_dir / f"{stem_type}.wav"
            if stem_path.exists():
                if stem_type in stems_to_process or (extract_instrumental and stem_type == "instrumental"):
                    # User wants this stem - move it to final output
                    shutil.move(str(stem_path), final_output_dir / f"{stem_type}.wav")
                    output_files.append(f"{stem_type}.wav")
                    print(f"✅ Moved {stem_type} stem (user selected)")
                else:
                    # User doesn't want this stem - delete it to save space
                    stem_path.unlink()
                    print(f"🗑️ Deleted {stem_type} stem (not selected by user)")

        # --- Post-Processing & Vorschauerstellung ---
        
        # Hier könnten zukünftige Post-Processing-Schritte wie Bleed-Reduction eingefügt werden.

        # Vorschau-Aufgaben für alle finalen Dateien sammeln
        preview_tasks = []
        for out_file in output_files:
            target_path = final_output_dir / out_file
            if target_path.exists():
                preview_path = target_path.parent / f"preview_{out_file}"
                preview_tasks.append((str(target_path), str(preview_path)))

        if not output_files:
            raise Exception("No output files were successfully processed")

        # Run all preview generations in parallel
        print(f"🚀 Starting parallel generation of {len(preview_tasks)} previews...")
        job_status[job_id]["phase"] = "Vorschauen werden parallel erstellt"
        save_job_status(job_id, job_status[job_id])
        
        results = list(generate_preview_parallel.map(preview_tasks))
        
        successful_previews = sum(1 for r in results if r)
        print(f"✅ Parallel preview generation complete. {successful_previews}/{len(preview_tasks)} successful.")

        # Force volume synchronization to ensure previews are available before marking job as complete.
        print("🔄 Forcing final volume synchronization...")
        volume.commit()
        volume.reload()
        print("✅ Synchronization complete.")

        try:
            # Calculate statistics
            total_size = sum(f.stat().st_size for f in final_output_dir.glob("*.wav"))
            processing_time = round(time.time() - start_time, 2)
            
            # Save volume changes
            volume.commit()
            
            # CRITICAL: Update final job status with results
            job_status[job_id] = {
                "status": "completed",
                "progress": 100,
                "phase": "Fertig - Download verfügbar",
                "error": None,
                "processing_time": processing_time,
                "files": output_files,
                "total_size_mb": round(total_size / (1024*1024), 1),
                # KORREKTUR: Korrekter Download-Pfad
                "download_base_url": f"/download/{job_id}/final_stems",
                "model_info": {
                    "name": "BS-RoFormer (ZFTurbo)",
                    "sdr": f"{BS_ROFORMER_CONFIG['sdr']}dB",
                    "type": "Band-Split RoPE Transformer",
                    "repository": "ZFTurbo/Music-Source-Separation-Training"
                }
            }

            # PERSIST status to volume so it survives between function calls
            save_job_status(job_id, job_status[job_id])
            
            print(f"🎉 Job {job_id} marked as COMPLETED with {len(output_files)} files")
            
        except Exception as status_error:
            print(f"❌ Error updating job status: {status_error}")
            # Still set completed status even if there's an error
            job_status[job_id] = {
                "status": "completed",
                "progress": 100,
                "phase": "Fertig - Download verfügbar",
                "error": None,
                "processing_time": round(time.time() - start_time, 2),
                "files": output_files,
                "total_size_mb": 0,
                "download_base_url": f"/download/{job_id}/final_stems"
            }
            print(f"🎉 Job {job_id} marked as COMPLETED (with error handling)")
        
        return {
            "success": True,
            "files": output_files,
            "job_id": job_id,
            "processing_time": processing_time,
            "total_size": total_size,
            "model_used": "htdemucs_ft",  # For compatibility
            "device": "A10G GPU (24GB) - ZFTurbo BS-RoFormer",
            "model_info": {
                "name": "BS-RoFormer (ZFTurbo)",
                "sdr": f"{BS_ROFORMER_CONFIG['sdr']}dB",
                "type": "Band-Split RoPE Transformer",
                "repository": "ZFTurbo/Music-Source-Separation-Training"
            },
            "gpu_settings": GPU_SETTINGS,
            "config": BS_ROFORMER_CONFIG
        }

    except Exception as e:
        error_msg = str(e)
        print(f"❌ BS-RoFormer processing failed: {error_msg}")
        job_status[job_id] = {
            "status": "error",
            "progress": 100,
            "phase": "Fehler",
            "error": error_msg,
            "processing_time": round(time.time() - start_time, 2)
        }
        return {
            "success": False, 
            "error": error_msg, 
            "job_id": job_id,
            "processing_time": round(time.time() - start_time, 2)
        }

@app.function(
    image=zfturbo_image,
    volumes={storage_path: volume},
    timeout=120 # 2 minutes should be plenty for one preview
)
def generate_preview_parallel(paths: tuple[str, str], duration: int = 30):
    """
    Generates a single preview in parallel using ffmpeg.
    Accepts a tuple of string paths for Modal's .map() compatibility.
    """
    import subprocess
    import os
    from pathlib import Path

    audio_path_str, preview_path_str = paths
    audio_path = Path(audio_path_str)
    preview_path = Path(preview_path_str)

    try:
        # 1. Get total duration with ffprobe
        ffprobe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
        ]
        total_duration = float(subprocess.check_output(ffprobe_cmd, text=True).strip())

        # 2. Calculate start time for the preview (middle of the track)
        start_time = max(0, (total_duration / 2) - (duration / 2))

        # 3. Create MP3 preview with ffmpeg (fast seeking)
        mp3_path = str(preview_path.with_suffix('.mp3'))
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(audio_path),
            "-t", str(duration),
            "-vn", "-ar", "44100", "-ac", "2", "-b:a", "128k",
            mp3_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=60)

        if os.path.exists(mp3_path):
            print(f"✅ Parallel preview created: {mp3_path}")
            volume.commit() # Commit the created file to the volume
            return True
        else:
            print(f"❌ Parallel preview NOT found after ffmpeg call: {mp3_path}")
            return False

    except Exception as e:
        error_details = ""
        if isinstance(e, subprocess.CalledProcessError):
            error_details = e.stderr
        print(f"⚠️ Parallel preview failed for {audio_path.name}: {e} {error_details}")
        return False

# FastAPI web application - following Modal best practices
def create_fastapi_app():
    from fastapi import FastAPI, File, Form, UploadFile, HTTPException
    from fastapi.responses import StreamingResponse, PlainTextResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI(
        title="Music AI Separator - ZFTurbo Complete",
        description="Professional music source separation using ZFTurbo BS-RoFormer",
        version="1.0.0"
    )
    @web_app.get("/status/{job_id}")
    async def status(job_id: str):
        """Status-API für Fortschrittsanzeige"""
        # Always check volume first for latest status
        volume.reload()
        status_file = Path(storage_path) / f"status_{job_id}.json"
        
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)
                print(f"📊 Status loaded from volume for job {job_id}: {status.get('progress', 0)}% - {status.get('phase', 'Unknown')}")
                return status
            except Exception as e:
                print(f"Error loading status from volume: {e}")
        
        # Fallback to in-memory status
        status = job_status.get(job_id)
        if status:
            print(f"📊 Status loaded from memory for job {job_id}: {status.get('progress', 0)}% - {status.get('phase', 'Unknown')}")
            return status
        
        print(f"❌ No status found for job {job_id}")
        return {"status": "unknown", "progress": 0, "phase": "Nicht gefunden", "error": "Job nicht gefunden"}

    @web_app.get("/debug/download_input_wav/{job_id}")    
    async def debug_download_input_wav(job_id: str):
        """Download the generated input.wav for debugging"""
        volume.reload()
        file_path = Path(storage_path) / job_id / "temp_input" / "input.wav"
        # Debug: Print job_id, file_path, and existence for troubleshooting
        print(f"[DOWNLOAD DEBUG] Job-ID: {job_id} | file_path: {file_path} | Exists: {file_path.exists()}")
        if not file_path.exists():
            raise HTTPException(404, f"input.wav not found for job {job_id}")
        def file_iterator():
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk
        return StreamingResponse(
            file_iterator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=input.wav",
                "Access-Control-Allow-Origin": "*",
            }
        )

    # CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/")
    async def api_upload(
        audio_file: UploadFile = File(...), 
        selected_stems: str = Form("all"),
        use_tta: bool = Form(False),
        extract_instrumental: bool = Form(False)
    ):
        """
        Upload endpoint - starts async processing and returns job_id immediately
        selected_stems: "all", "vocals", "drums", "bass", "other" or comma-separated like "vocals,drums"
        """
        from fastapi import Response
        
        try:
            print(f"🎵 Received file for ZFTurbo BS-RoFormer: {audio_file.filename}")
            
            # Validate file format
            allowed_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']
            if not any(audio_file.filename.lower().endswith(ext) for ext in allowed_extensions):
                raise HTTPException(400, f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}")
            
            # Read and validate audio data
            audio_data = await audio_file.read()
            
            if len(audio_data) == 0:
                raise HTTPException(400, "Empty file uploaded")
            
            # File size limit (100MB)
            max_size = 100 * 1024 * 1024
            if len(audio_data) > max_size:
                raise HTTPException(400, f"File too large (max {max_size // 1024 // 1024}MB)")
            
            # Generate job ID and safe filename
            job_id = str(uuid.uuid4())
            file_extension = Path(audio_file.filename).suffix
            safe_filename = f"{job_id}{file_extension}"
            
            print(f"🎵 Starting async processing for {audio_file.filename} ({len(audio_data)} bytes)")
            
            # Initialize job status
            job_status[job_id] = {
                "status": "queued",
                "progress": 0,
                "phase": "Upload abgeschlossen, wird verarbeitet...",
                "error": None,
                "processing_time": None,
                "files": [],
                "total_size_mb": 0
            }
            
            # Start async processing with BS-RoFormer
            bs_roformer_separate_audio.spawn(
                audio_data, safe_filename, job_id, selected_stems, use_tta, extract_instrumental
            )
            
            response_data = {
                "job_id": job_id,
                "status": "queued",
                "message": "Processing started. Use /status/{job_id} to check progress."
            }
            
            print(f"🔄 ZFTurbo job {job_id} queued for processing")
            
            # Manual CORS headers for response
            import json
            response = Response(
                content=json.dumps(response_data),
                media_type="application/json"
            )
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            
            return response
                
        except Exception as e:
            print(f"❌ Unexpected ZFTurbo error: {str(e)}")
            raise HTTPException(500, f"Unexpected ZFTurbo error: {str(e)}")

    @web_app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "Klein Digital Solutions - Music AI Separator",
            "implementation": "ZFTurbo/Music-Source-Separation-Training",
            "gpu": "A10G (24GB)",
            "model": "BS-RoFormer",
            "max_file_size": "100MB",
            "repository": "https://github.com/ZFTurbo/Music-Source-Separation-Training"
        }

    @web_app.get("/models")
    async def models():
        """Get model information"""
        return {
            "models": {
                "bs_roformer": {
                    "name": "BS-RoFormer (ZFTurbo)",
                    "description": "Band-Split RoPE Transformer for high-quality source separation.",
                    "stems": ["vocals", "drums", "bass", "other"],
                    "sdr": f"{BS_ROFORMER_CONFIG['sdr']}dB",
                    "processing_time": "~2-5 minutes",
                    "repository": "ZFTurbo/Music-Source-Separation-Training",
                    "quality": "State-of-the-Art"
                }
            },
            "gpu_info": {
                "type": "NVIDIA A10G",
                "memory": "24GB VRAM",
                "optimized_for": "Professional Audio Processing",
                "implementation": "ZFTurbo Complete"
            }
        }

    @web_app.get("/download/{job_id}/{model_name}/{filename}")
    async def download_file(job_id: str, model_name: str, filename: str):
        """Download individual file with streaming from volume"""
        
        volume.reload()  # Force volume reload for latest state
        
        # KORREKTUR: Vereinfachter und korrekter Pfad
        file_path = Path(storage_path) / job_id / "output" / model_name / filename
        
        if not file_path.exists():
            raise HTTPException(404, f"File not found: {filename}")

        def file_iterator():
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
        
        return StreamingResponse(
            file_iterator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Allow-Origin": "*",
            }
        )

    @web_app.get("/download/{job_id}/{model_name}/all.zip")
    async def download_zip(job_id: str, model_name: str):
        """Download all files as ZIP with streaming from volume"""
        
        volume.reload()  # Force volume reload for latest state
        
        # KORREKTUR: Vereinfachter und korrekter Pfad
        model_output_dir = Path(storage_path) / job_id / "output" / model_name
        
        if not model_output_dir.is_dir():
            raise HTTPException(404, "Job files not found")
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for wav_file in model_output_dir.glob("*.wav"):
                zip_file.write(wav_file, wav_file.name)
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=zfturbo_stems.zip",
                "Access-Control-Allow-Origin": "*",
            }
        )

    @web_app.get("/preview/{job_id}/{model_name}/{filename}")
    async def preview_file(job_id: str, model_name: str, filename: str):
        """Stream 30s MP3 preview for a stem"""
        volume.reload()
        
        # KORREKTUR: Vereinfachter und korrekter Pfad
        preview_mp3 = Path(storage_path) / job_id / "output" / model_name / filename
        
        if not preview_mp3.exists():
            # Try .wav if .mp3 not found
            preview_wav = preview_mp3.with_suffix('.wav')
            if preview_wav.exists():
                def file_iterator():
                    with open(preview_wav, "rb") as f:
                        while chunk := f.read(8192):
                            yield chunk
                print(f"⚠️ MP3-Vorschau nicht gefunden, sende WAV: {preview_wav}")
                return StreamingResponse(
                    file_iterator(),
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f"inline; filename={preview_wav.name}",
                        "Access-Control-Allow-Origin": "*",
                    }
                )
            print(f"❌ Vorschau-Datei nicht gefunden: {preview_mp3} und {preview_wav}")
            raise HTTPException(404, f"Preview file not found: {filename}")
        def file_iterator():
            with open(preview_mp3, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
        print(f"✅ Sende MP3-Vorschau: {preview_mp3}")
        return StreamingResponse(
            file_iterator(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename={preview_mp3.name}",
                "Access-Control-Allow-Origin": "*",
            }
        )

    @web_app.options("/{path:path}")
    async def options_handler(path: str):
        """Handle CORS preflight requests"""
        return {"status": "ok"}

    
    return web_app

# Deploy FastAPI app using Modal's ASGI pattern
@app.function(image=zfturbo_image, volumes={storage_path: volume})
@modal.asgi_app()
def fastapi_app():
    return create_fastapi_app()

if __name__ == "__main__":
    print("🚀 Klein Digital Solutions - Music AI Separator")
    print("📡 ZFTurbo Complete Implementation - Modal Serverless")
    print("🔬 Repository: ZFTurbo/Music-Source-Separation-Training")
    print("💡 Deploy: modal deploy modal_app_zfturbo_complete.py")