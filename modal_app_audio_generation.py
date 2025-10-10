#!/usr/bin/env python3
"""
Klein Digital Solutions - AudioLDM Text-to-Audio Generation
Modal Serverless Deployment with A10G GPU - Production Ready

Features:
- AudioLDM-L-Full (975M params) - Best Quality
- Text-to-Audio Generation
- Multiple audio formats (WAV, MP3)
- Comprehensive error handling
- Modal volume file handling
"""

import modal
import os
from pathlib import Path
import tempfile
import uuid
import time
import json
import io
import torch
import torchaudio
import numpy as np

# Global storage for temporary files
temp_files_storage = {}
# Status dictionary for job progress
job_status = {}

# Modal app instance
app = modal.App("audio-generation-audioldm2-music")

# Create persistent volume for file storage
volume = modal.Volume.from_name("audio-gen-files", create_if_missing=True)
storage_path = "/vol/audio_files"

# AudioLDM-L-Full requirements implementation
audioldm_image = modal.Image.debian_slim(python_version="3.11").apt_install([
    "ffmpeg",
    "libsndfile1", 
    "git",
    "build-essential",
    "python3-dev"
]).pip_install([
    # Core PyTorch stack
    "torch>=2.0.1",
    "torchaudio",
    "torchvision",
    
    # Hugging Face stack - AudioLDM2 kompatible Versionen (Fix für cache_position)
    "diffusers>=0.30.0",
    "transformers>=4.43.0",
    "accelerate",
    "huggingface-hub",
    
    # Audio processing
    "scipy",
    "soundfile",
    "librosa",
    "numpy",
    
    # FastAPI stack
    "fastapi>=0.100.0,<0.104.0",
    "python-multipart>=0.0.6",
    
    # Additional packages
    "requests",
    "Pillow"
]).run_commands([
    # Pre-download model to reduce cold start time
    "python -c \"import torch; from diffusers import AudioLDMPipeline; AudioLDMPipeline.from_pretrained('cvssp/audioldm-l-full', torch_dtype=torch.float16)\""
])

# AudioLDM-L-Full Configuration (975M Parameter - Höchste Qualität)
AUDIO_CONFIG = {
    "model_id": "cvssp/audioldm-l-full",
    "parameters": "975M",
    "max_audio_length": 30.0,  # seconds  
    "sample_rate": 16000,  # 16kHz
    "quality": "Highest Quality - Large Model",
    "license": "CreativeML Open RAIL-M (Commercial Use OK)"
}

# GPU Memory Optimization Settings
GPU_SETTINGS = {
    "device_ids": "0",
    "use_cuda": True,
    "memory_efficient": True,
    "precision": "float16"  # For memory efficiency with large model
}

def save_job_status(job_id: str, status_data: dict):
    """Save job status to volume for persistence"""
    try:
        status_file = Path(storage_path) / f"status_{job_id}.json"
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
        volume.commit()
        print(f"📊 Audio generation status saved for job {job_id}: {status_data.get('progress', 0)}% - {status_data.get('phase', 'Unknown')}")
    except Exception as e:
        print(f"⚠️ Error saving job status: {e}")

@app.function(
    image=audioldm_image,
    gpu="a10g", 
    timeout=1200,  # 20 minutes for large model
    memory=20480,  # 20GB memory for 975M model
    volumes={storage_path: volume},
    scaledown_window=300  # Keep warm longer for large model
)
def generate_audio_from_text(
    prompt: str,
    job_id: str,
    audio_length: float = 10.0,
    num_inference_steps: int = 100,
    guidance_scale: float = 6.0,
    output_format: str = "wav"
) -> dict:
    """
    AudioLDM-L-Full Text-to-Audio Generation
    Using the largest, highest quality model (975M parameters)
    """
    start_time = time.time()
    
    # Set initial job status
    job_status[job_id] = {
        "status": "initializing",
        "progress": 5,
        "phase": "AudioLDM2-Music wird geladen",
        "error": None,
        "processing_time": None
    }
    save_job_status(job_id, job_status[job_id])
    
    try:
        # Setup directories
        job_dir = Path(storage_path) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = job_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        print(f"🎵 Starting AudioLDM-L-Full generation for job {job_id}")
        print(f"📝 Prompt: {prompt}")
        print(f"⏱️ Duration: {audio_length}s")
        
        # Update status - Loading model
        job_status[job_id] = {
            "status": "loading",
            "progress": 20,
            "phase": "AudioLDM2-Music Modell wird geladen (350M Parameter)",
            "error": None,
            "processing_time": None
        }
        save_job_status(job_id, job_status[job_id])
        
        # Load AudioLDM pipeline (Stabil und funktioniert)
        from diffusers import AudioLDMPipeline
        
        pipe = AudioLDMPipeline.from_pretrained(
            AUDIO_CONFIG["model_id"],  # AudioLDM-L-Full 975M Parameter
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            print("✅ AudioLDM loaded on GPU")
        else:
            print("⚠️ AudioLDM loaded on CPU (slower)")
        
        # Update status - Generating
        job_status[job_id] = {
            "status": "generating",
            "progress": 40,
            "phase": "Audio wird generiert mit AudioLDM2-Music",
            "error": None,
            "processing_time": None
        }
        save_job_status(job_id, job_status[job_id])
        
        # Generate audio
        print(f"🎨 Generating audio with {num_inference_steps} steps...")
        
        # Create generator for reproducible results
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(42)
        
        audio_output = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length,
            guidance_scale=guidance_scale,
            num_waveforms_per_prompt=1,
            generator=generator
        )
        
        # Extract audio array
        audio_array = audio_output.audios[0]  # Shape: [length] for AudioLDM-L-Full
        sample_rate = AUDIO_CONFIG["sample_rate"]  # 16kHz
        
        print(f"✅ Audio generated with AudioLDM2-Music - Shape: {audio_array.shape}, Sample Rate: {sample_rate}")
        
        # Update status - Saving
        job_status[job_id] = {
            "status": "saving",
            "progress": 80,
            "phase": "Audio wird gespeichert",
            "error": None,
            "processing_time": None
        }
        save_job_status(job_id, job_status[job_id])
        
        # Save audio files
        output_files = []
        
        # Save WAV (always)
        wav_path = output_dir / f"generated_audio.wav"
        # Ensure correct tensor shape for torchaudio
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.from_numpy(audio_array)
        else:
            audio_tensor = audio_array
        
        # Ensure 2D tensor: [channels, samples]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
        
        torchaudio.save(str(wav_path), audio_tensor, sample_rate)
        output_files.append("generated_audio.wav")
        print(f"💾 WAV saved: {wav_path}")
        
        # Convert to MP3 if requested
        if output_format == "mp3" or output_format == "both":
            mp3_path = output_dir / f"generated_audio.mp3"
            
            # Use the already prepared audio_tensor
            torchaudio.save(str(mp3_path), audio_tensor, sample_rate, format="mp3")
            output_files.append("generated_audio.mp3")
            print(f"💾 MP3 saved: {mp3_path}")
        
        # Generate preview (30-second version for quick testing)
        preview_length = min(30.0, audio_length)
        if audio_length > 30.0:
            preview_samples = int(preview_length * sample_rate)
            
            # Create preview from audio_tensor
            if audio_tensor.dim() == 2:
                preview_tensor = audio_tensor[:, :preview_samples]  # [channels, samples]
            else:
                preview_tensor = audio_tensor[:preview_samples].unsqueeze(0)  # [1, samples]
            
            preview_path = output_dir / f"preview_audio.wav"
            torchaudio.save(str(preview_path), preview_tensor, sample_rate)
            output_files.append("preview_audio.wav")
            print(f"🎧 Preview saved: {preview_path}")
        
        # Calculate statistics
        processing_time = round(time.time() - start_time, 2)
        total_size = sum(f.stat().st_size for f in output_dir.glob("*"))
        
        # Save volume changes
        volume.commit()
        
        # Final job status
        job_status[job_id] = {
            "status": "completed",
            "progress": 100,
            "phase": "Fertig - Audio generiert",
            "error": None,
            "processing_time": processing_time,
            "files": output_files,
            "total_size_mb": round(total_size / (1024*1024), 1),
            "download_base_url": f"/download/{job_id}/output",
            "audio_info": {
                "duration": audio_length,
                "sample_rate": sample_rate,
                "format": output_format,
                "prompt": prompt
            },
            "model_info": {
                "name": "AudioLDM2-Music",
                "parameters": AUDIO_CONFIG["parameters"],
                "quality": AUDIO_CONFIG["quality"],
                "inference_steps": num_inference_steps
            }
        }
        
        save_job_status(job_id, job_status[job_id])
        
        print(f"🎉 AudioLDM2-Music generation completed for job {job_id}")
        print(f"📊 Files: {len(output_files)}, Size: {job_status[job_id]['total_size_mb']}MB")
        
        return {
            "success": True,
            "files": output_files,
            "job_id": job_id,
            "processing_time": processing_time,
            "audio_duration": audio_length,
            "model_used": "AudioLDM2-Music",
            "device": "A10G GPU (24GB) - AudioLDM2-Music 350M",
            "sample_rate": sample_rate,
            "prompt": prompt
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ AudioLDM2-Music generation failed: {error_msg}")
        job_status[job_id] = {
            "status": "error",
            "progress": 100,
            "phase": "Fehler bei Audio-Generierung",
            "error": error_msg,
            "processing_time": round(time.time() - start_time, 2)
        }
        save_job_status(job_id, job_status[job_id])
        return {
            "success": False, 
            "error": error_msg, 
            "job_id": job_id,
            "processing_time": round(time.time() - start_time, 2)
        }

# FastAPI web application
def create_fastapi_app():
    from fastapi import FastAPI, Form, HTTPException
    from fastapi.responses import StreamingResponse, PlainTextResponse
    from fastapi.middleware.cors import CORSMiddleware
    
    web_app = FastAPI(
        title="AudioLDM2-Music Text-to-Audio Generation",
        description="Professional music generation using AudioLDM2-Music (350M parameters, music-optimized)",
        version="2.0.0"
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
    async def generate_audio(
        prompt: str = Form(...),
        audio_length: float = Form(10.0),
        num_inference_steps: int = Form(100),
        guidance_scale: float = Form(6.0),
        output_format: str = Form("wav")
    ):
        """Generate music from text prompt using AudioLDM2-Music"""
        try:
            # Validate inputs
            if len(prompt.strip()) == 0:
                raise HTTPException(400, "Prompt cannot be empty")
            
            if not (5.0 <= audio_length <= 30.0):
                raise HTTPException(400, "Audio length must be between 5 and 30 seconds")
            
            if not (10 <= num_inference_steps <= 100):
                raise HTTPException(400, "Inference steps must be between 10 and 100")
            
            if output_format not in ["wav", "mp3", "both"]:
                raise HTTPException(400, "Output format must be 'wav', 'mp3', or 'both'")
            
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            print(f"🎵 Starting AudioLDM2-Music generation for prompt: {prompt}")
            
            # Start async processing
            generate_audio_from_text.spawn(
                prompt, job_id, audio_length, num_inference_steps, guidance_scale, output_format
            )
            
            return {
                "job_id": job_id,
                "status": "started",
                "message": f"AudioLDM2-Music generation started for: {prompt[:50]}...",
                "expected_duration": f"{audio_length}s audio"
            }
            
        except Exception as e:
            print(f"❌ API error: {str(e)}")
            raise HTTPException(500, f"Generation failed: {str(e)}")
    
    @web_app.get("/status/{job_id}")
    async def get_status(job_id: str):
        """Get generation status"""
        volume.reload()
        status_file = Path(storage_path) / f"status_{job_id}.json"
        
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)
                return status
            except Exception as e:
                print(f"Error loading status: {e}")
        
        # Fallback to in-memory status
        status = job_status.get(job_id)
        if status:
            return status
        
        return {"status": "unknown", "progress": 0, "phase": "Job nicht gefunden", "error": "Job ID not found"}
    
    @web_app.get("/download/{job_id}/output/{filename}")
    async def download_file(job_id: str, filename: str):
        """Download generated audio file"""
        volume.reload()
        file_path = Path(storage_path) / job_id / "output" / filename
        
        if not file_path.exists():
            raise HTTPException(404, f"File not found: {filename}")
        
        def file_iterator():
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
        
        # Determine media type based on file extension
        media_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        
        return StreamingResponse(
            file_iterator(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Allow-Origin": "*",
            }
        )
    
    @web_app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "AudioLDM2-Music Text-to-Audio Generation",
            "model": "AudioLDM2-Music (350M parameters, music-optimized)",
            "gpu": "A10G (24GB)",
            "capabilities": ["text-to-audio", "wav", "mp3"],
            "max_audio_length": "20 seconds",
            "sample_rate": "16kHz",
            "specialization": "Music Production",
            "license": "Open RAIL-M (Commercial OK)"
        }
    
    return web_app

# Deploy FastAPI app
@app.function(image=audioldm_image, volumes={storage_path: volume})
@modal.asgi_app()
def fastapi_app():
    return create_fastapi_app()

if __name__ == "__main__":
    print("🎵 Klein Digital Solutions - AudioLDM2-Music Text-to-Audio Generation")
    print("🎨 AudioLDM2-Music (350M Parameters, Music-Optimized) - Modal Serverless")
    print("💡 Deploy: modal deploy modal_app_audio_generation.py")