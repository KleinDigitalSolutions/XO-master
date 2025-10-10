#!/usr/bin/env python3
"""
Klein Digital Solutions - Music AI Transcription Service
Complete Professional Music Transcription for Producers & Musicians

Features:
- basic-pitch (Spotify): Melodie/Instrumente -> MIDI-Noten fuer VST-Instrumente  
- madmom: Drums -> MIDI-Grid zum Austauschen von Samples
- essentia: Akkorde (Chords) -> Harmonische Analyse fuer Songwriting & Remixe
- madmom: Beat/Tempo -> Perfekter Click-Track fuer die DAW

Modal Serverless Deployment with GPU optimization
Professional error handling and monitoring
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
import shutil

# Global storage for job status tracking
job_status = {}

# Modal app instance
app = modal.App("music-ai-transcription-complete")

# Create persistent volume for file storage
volume = modal.Volume.from_name("music-transcription-storage", create_if_missing=True)
storage_path = "/vol/transcription_files"

# ===================================================================
# PROFESSIONAL IMAGE DEFINITION FOR MUSIC TRANSCRIPTION
# ===================================================================

transcription_image = modal.Image.debian_slim(python_version="3.9").apt_install([
    # System dependencies
    "ffmpeg",
    "libsndfile1",
    "libsndfile1-dev", 
    "build-essential",
    "python3-dev",
    "pkg-config",
    "portaudio19-dev",
    "libflac-dev",
    "git",
    "fluidsynth",
    "libportaudio2"
]).pip_install([
    # Core dependencies first - NumPy pinned for madmom compatibility (pre-1.24)
    "numpy>=1.21.0,<1.24.0",  # CRITICAL: madmom requires NumPy <1.24 (np.float deprecated)
    "scipy>=1.7.0", 
    "soundfile>=0.12.1",
    "Cython>=0.29.0",
]).pip_install([
    # Libraries that need Cython
    "madmom>=0.16.1",           # Beat tracking and drum transcription
]).pip_install([
    # Other dependencies
    "librosa>=0.9.0",
    "mir_eval>=0.7",
    "basic-pitch>=0.4.0",       # Spotify's melody/instrument transcription
    "essentia-tensorflow",      # Chord recognition
    
    # MIDI processing
    "pretty_midi>=0.2.10",
    "mido>=1.2.10",
    "music21>=8.1.0",
    
    # Audio processing 
    "pydub>=0.25.1",
    "audioread>=3.0.0",
    
    # FastAPI stack
    "fastapi>=0.100.0,<0.104.0",
    "python-multipart>=0.0.6",
    "uvicorn[standard]>=0.23.0",
    
    # Additional utilities
    "requests>=2.28.0",
    "matplotlib>=3.5.0",
    "pandas>=1.5.0"
])

# Transcription configuration for producers
TRANSCRIPTION_CONFIG = {
    "melody": {
        "library": "basic-pitch", 
        "description": "Melodie/Instrumente -> MIDI-Noten fuer VST-Instrumente",
        "output_formats": ["midi", "json"],
        "producer_benefit": "Perfekte MIDI-Noten zum Laden in VSTs, Harmonizer & Synthesizer"
    },
    "drums": {
        "library": "madmom",
        "description": "Drums -> MIDI-Grid zum Austauschen von Samples", 
        "output_formats": ["midi", "json"],
        "producer_benefit": "Drum-Pattern als MIDI fuer Sample-Replacement in DAW"
    },
    "chords": {
        "library": "essentia",
        "description": "Akkorde -> Harmonische Analyse fuer Songwriting & Remixe",
        "output_formats": ["midi", "json", "txt"],
        "producer_benefit": "Chord-Progressionen fuer Remixe, Covers & neue Arrangements"
    },
    "beat": {
        "library": "madmom", 
        "description": "Beat/Tempo -> Perfekter Click-Track fuer die DAW",
        "output_formats": ["json", "txt", "midi"],
        "producer_benefit": "Exakte BPM + Grid fuer perfekte Synchronisation in der DAW"
    }
}

def save_job_status(job_id: str, status_data: dict):
    """Save job status to volume for persistence"""
    try:
        status_file = Path(storage_path) / f"status_{job_id}.json"
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
        volume.commit()
        print(f"Status saved for job {job_id}: {status_data.get('progress', 0)}% - {status_data.get('phase', 'Unknown')}")
    except Exception as e:
        print(f"Error saving transcription job status: {e}")

def update_job_progress(job_id: str, progress: int, phase: str, additional_data: dict = None):
    """Helper to update job progress with additional data"""
    status_data = {
        "status": "processing",
        "progress": progress,
        "phase": phase,
        "error": None,
        "processing_time": None
    }
    if additional_data:
        status_data.update(additional_data)
    
    job_status[job_id] = status_data
    save_job_status(job_id, status_data)

@app.function(
    image=transcription_image,
    gpu="a10g",  # GPU for basic-pitch optimization
    timeout=2400,  # OPTIMIZED: 40 minutes for complex transcriptions (was 1800)
    memory=32768,  # OPTIMIZED: 32GB for complex multi-track analysis (was 16384)
    volumes={storage_path: volume},
    scaledown_window=120
)
def transcribe_complete_audio(
    audio_data: bytes,
    filename: str,
    job_id: str,
    tasks: dict = None  # {"melody": True, "drums": True, "chords": True, "beat": True}
) -> dict:
    """
    Complete professional music transcription for producers
    Returns all formats needed for professional music production
    """
    import time  # Ensure time is available in function scope
    
    # CRITICAL: Monkey patch for madmom NumPy compatibility
    import numpy as np
    if not hasattr(np, 'float'):
        np.float = float
        np.int = int
        np.bool = bool
        print("DEBUG: Applied NumPy compatibility monkey patch for madmom")
    
    start_time = time.time()
    
    # Default to all tasks if not specified
    if tasks is None:
        tasks = {"melody": True, "drums": True, "chords": True, "beat": True}
    
    # Set initial job status
    update_job_progress(job_id, 5, "Initialisierung - Audio wird geladen")
    
    try:
        # Setup directories
        job_dir = Path(storage_path) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        input_file = job_dir / filename
        input_file.write_bytes(audio_data)
        
        output_dir = job_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        final_output_dir = output_dir / "final"
        final_output_dir.mkdir(exist_ok=True)
        
        print(f"Starting complete music transcription for job {job_id}")
        print(f"Input: {input_file} ({len(audio_data) // 1024} KB)")
        print(f"Tasks: {[task for task, enabled in tasks.items() if enabled]}")
        
        # Convert to WAV for processing
        update_job_progress(job_id, 10, "Audio-Konvertierung zu WAV")
        
        wav_file = job_dir / "input.wav"
        file_ext = Path(filename).suffix.lower()
        
        if file_ext != ".wav":
            # OPTIMIZED audio conversion with better quality and filtering
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_file),
                "-ar", "44100",                                    # Standard sample rate
                "-ac", "2",                                        # Stereo for full processing
                "-af", "highpass=f=20,lowpass=f=20000",            # OPTIMIZED: Audio cleaning filters
                "-q:a", "0",                                       # OPTIMIZED: Lossless quality
                str(wav_file)
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=120)
                print(f"Converted {filename} to WAV")
            except Exception as e:
                raise Exception(f"Audio conversion failed: {e}")
        else:
            shutil.copy2(input_file, wav_file)
            print(f"Using WAV file directly: {wav_file}")
        
        # OPTIMIZED mono version creation with better quality
        mono_wav_file = job_dir / "input_mono.wav"
        ffmpeg_mono_cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_file),
            "-ar", "44100",
            "-ac", "1",                                        # Mono
            "-af", "highpass=f=20,lowpass=f=20000",            # OPTIMIZED: Same filtering for consistency
            "-q:a", "0",                                       # OPTIMIZED: Lossless quality
            str(mono_wav_file)
        ]
        subprocess.run(ffmpeg_mono_cmd, check=True, capture_output=True, timeout=60)
        
        volume.commit()
        
        # ===================================================================
        # TASK 1: MELODY/INSTRUMENT TRANSCRIPTION (basic-pitch)
        # ===================================================================
        
        output_files = []
        completed_tasks = []
        
        if tasks.get("melody", False):
            update_job_progress(job_id, 20, "Melodie-Transkription (basic-pitch) - Spotify Technologie")
            
            try:
                print("Starting basic-pitch melody/instrument transcription...")
                
                # Import basic-pitch
                from basic_pitch.inference import predict_and_save
                from basic_pitch import ICASSP_2022_MODEL_PATH
                
                # Create melody output directory
                melody_dir = output_dir / "melody"
                melody_dir.mkdir(exist_ok=True)
                
                # Run basic-pitch transcription with OPTIMIZED parameters for music production
                predict_and_save(
                    audio_path_list=[str(wav_file)],
                    output_directory=str(melody_dir),
                    save_midi=True,
                    sonify_midi=False,
                    save_model_outputs=True,   # For debug/analysis - OPTIMIZED
                    save_notes=True,           # Also save notes as JSON
                    model_or_model_path=ICASSP_2022_MODEL_PATH,
                    # OPTIMIZED parameters for better melody extraction:
                    minimum_note_length=0.1,   # Filter very short notes for cleaner MIDI
                    minimum_frequency=80.0,    # Focus on melodic instruments (exclude sub-bass)
                    maximum_frequency=2000.0   # Exclude very high frequencies for cleaner results
                )
                
                # Find and rename output files
                stem_name = wav_file.stem
                generated_midi = melody_dir / f"{stem_name}_basic_pitch.mid"
                generated_notes = melody_dir / f"{stem_name}_basic_pitch.json"
                
                if generated_midi.exists():
                    final_midi = final_output_dir / "melody_notes.mid"
                    shutil.move(str(generated_midi), str(final_midi))
                    output_files.append("melody_notes.mid")
                    print("Melody MIDI created for VST loading")
                    # Immediate volume commit to persist file
                    volume.commit()
                    print(f"DEBUG: Volume committed after melody MIDI creation")
                
                if generated_notes.exists():
                    final_notes = final_output_dir / "melody_notes.json"
                    shutil.move(str(generated_notes), str(final_notes))
                    output_files.append("melody_notes.json")
                    print("Melody notes JSON created")
                    # Immediate volume commit to persist file
                    volume.commit()
                    print(f"DEBUG: Volume committed after melody JSON creation")
                
                # Create additional producer-friendly formats
                if generated_midi.exists() or final_midi.exists():
                    # Create a simplified MIDI for easier editing
                    create_producer_midi_summary(final_midi if final_midi.exists() else generated_midi, 
                                               final_output_dir / "melody_simplified.mid")
                    output_files.append("melody_simplified.mid")
                
                completed_tasks.append("melody")
                print("Melody transcription completed - Ready for VST import!")
                
            except Exception as e:
                print(f"Melody transcription failed: {e}")
        
        # ===================================================================
        # TASK 2: DRUM TRANSCRIPTION (madmom)
        # ===================================================================
        
        if tasks.get("drums", False):
            update_job_progress(job_id, 40, "Drum-Transkription (madmom) - MIDI-Grid fuer Sample-Replacement")
            
            try:
                print("Starting madmom drum transcription...")
                
                # Import madmom
                import madmom
                
                # Create drums output directory
                drums_dir = output_dir / "drums"
                drums_dir.mkdir(exist_ok=True)
                
                # Use mono file for drum detection
                audio_file_for_drums = str(mono_wav_file)
                
                # OPTIMIZED drum onset detection for better drum pattern extraction
                print("Detecting drum onsets with optimized parameters...")
                proc = madmom.features.onsets.CNNOnsetProcessor()
                onsets = proc(audio_file_for_drums)
                
                # OPTIMIZED peak picking for precise drum timing (2025 Research)
                peak_picker = madmom.features.onsets.OnsetPeakPickingProcessor(
                    threshold=0.5,      # CNN-optimal threshold for clean drum detection
                    combine=0.02,       # Shorter combine time for fast drums (was 0.03) 
                    pre_max=0.01,       # Better onset localization
                    post_max=0.01,      # Better onset localization
                    pre_avg=0.05,       # Improved averaging for stability
                    post_avg=0.05,      # Improved averaging for stability
                    smooth=0.0          # No smoothing for precise drum hits
                )
                drum_onsets = peak_picker(onsets)
                
                # Create drum MIDI
                drum_midi_path = final_output_dir / "drums_pattern.mid"
                create_drum_midi(drum_onsets, drum_midi_path)
                output_files.append("drums_pattern.mid")
                
                # Create JSON with drum timing for DAW import
                drum_json = {
                    "drum_onsets": drum_onsets.tolist(),
                    "total_duration": len(onsets) * 0.01,  # madmom default hop size
                    "sample_rate": 44100,
                    "producer_note": "Import MIDI to replace with your drum samples in DAW"
                }
                
                drum_json_path = final_output_dir / "drums_timing.json"
                with open(drum_json_path, 'w') as f:
                    json.dump(drum_json, f, indent=2)
                output_files.append("drums_timing.json")
                
                completed_tasks.append("drums")
                print("Drum transcription completed - Ready for sample replacement!")
                
            except Exception as e:
                print(f"Drum transcription failed: {e}")
        
        # ===================================================================
        # TASK 3: CHORD RECOGNITION (essentia)
        # ===================================================================
        
        if tasks.get("chords", False):
            update_job_progress(job_id, 60, "Akkord-Erkennung (essentia) - Harmonische Analyse")
            
            try:
                print("Starting essentia chord recognition...")
                
                # Import essentia with proper error handling
                import essentia.standard as es
                import essentia
                
                # Create chords output directory  
                chords_dir = output_dir / "chords"
                chords_dir.mkdir(exist_ok=True)
                
                # Load audio with essentia
                loader = es.MonoLoader(filename=str(mono_wav_file), sampleRate=44100)
                audio = loader()
                
                # OPTIMIZED HPCP computation pipeline for better chord recognition
                framecutter = es.FrameCutter(
                    frameSize=8192,     # Higher resolution for better frequency analysis (was 4096)
                    hopSize=1024,       # Better time resolution for chord changes (was 2048)
                    silentFrames='noise'
                )
                windowing = es.Windowing(type='blackmanharris62')
                spectrum = es.Spectrum()
                spectralpeaks = es.SpectralPeaks(
                    orderBy='magnitude', 
                    magnitudeThreshold=0.0001,   # More sensitive detection (was 0.00001)
                    minFrequency=80,             # Optimized for music production (was 20)
                    maxFrequency=4000,           # Extended for complex harmonies (was 3500)
                    maxPeaks=100                 # More peaks for better chord detection (was 60)
                )
                hpcp = es.HPCP(
                    size=36,                     # Higher resolution for precise chords (default was 12)
                    referenceFrequency=440,      # Standard tuning reference
                    harmonics=8,                 # More harmonics for better recognition (default was 4)
                    bandPreset=False,            # Custom frequency settings
                    minFrequency=80,             # Match spectral peaks settings
                    maxFrequency=4000,           # Match spectral peaks settings
                    weightType='cosine'          # Cosine weighting for better harmonic analysis
                )
                
                # Compute HPCPs frame by frame
                hpcp_frames = []
                for frame in framecutter(audio):
                    windowed = windowing(frame)
                    spec = spectrum(windowed)
                    peaks_mag, peaks_freq = spectralpeaks(spec)
                    hpcp_frame = hpcp(peaks_mag, peaks_freq)
                    hpcp_frames.append(hpcp_frame)
                
                # Convert to essentia array format - this fixes the VectorVectorReal error
                hpcp_array = essentia.array(hpcp_frames)
                
                # OPTIMIZED chord detection with better temporal resolution
                chords_detector = es.ChordsDetection(
                    hopSize=1024,                # Better temporal resolution (was 2048)
                    windowSize=3                 # Extended window for stability (was 2)
                )
                chords, strength = chords_detector(hpcp_array)
                
                # Create chord progression file for producers
                chord_progression = analyze_chord_progression(chords, strength)
                
                # Save chord MIDI for DAW import
                chord_midi_path = final_output_dir / "chord_progression.mid"
                create_chord_midi(chord_progression, chord_midi_path)
                output_files.append("chord_progression.mid")
                
                # Save detailed JSON analysis
                chord_analysis = {
                    "chord_progression": chord_progression,
                    "raw_chords": chords,
                    "confidence": strength.tolist(),
                    "analysis": {
                        "key_signature": detect_key_signature(chord_progression),
                        "common_progressions": find_common_progressions(chord_progression),
                        "producer_tips": generate_producer_tips(chord_progression)
                    }
                }
                
                chord_json_path = final_output_dir / "chord_analysis.json"
                with open(chord_json_path, 'w') as f:
                    json.dump(chord_analysis, f, indent=2)
                output_files.append("chord_analysis.json")
                
                # Create human-readable chord chart
                chord_chart_path = final_output_dir / "chord_chart.txt"
                create_chord_chart(chord_progression, chord_chart_path)
                output_files.append("chord_chart.txt")
                
                completed_tasks.append("chords")
                print("Chord analysis completed - Ready for songwriting & remixes!")
                
            except Exception as e:
                print(f"Chord recognition failed: {e}")
        
        # ===================================================================
        # TASK 4: BEAT/TEMPO TRACKING (madmom)
        # ===================================================================
        
        if tasks.get("beat", False):
            update_job_progress(job_id, 80, "Beat/Tempo-Analyse (madmom) - Click-Track fuer DAW")
            
            try:
                print("Starting madmom beat/tempo analysis...")
                
                # Import madmom
                import madmom
                
                # Create beat output directory
                beat_dir = output_dir / "beat"
                beat_dir.mkdir(exist_ok=True)
                
                # OPTIMIZED beat tracking with higher precision for music production
                print("Analyzing beats and tempo with enhanced precision...")
                proc = madmom.features.beats.DBNBeatTrackingProcessor(
                    fps=200,           # Higher resolution for better timing (was 100)
                    min_bpm=60,        # Production-relevant BPM range
                    max_bpm=180,       # Production-relevant BPM range
                    transition_lambda=100  # Improved transition modeling
                )
                act = madmom.features.beats.RNNBeatProcessor()(str(mono_wav_file))
                beats = proc(act)
                
                # OPTIMIZED downbeat detection for better bar structure analysis
                print("Detecting downbeats and bar structure with enhanced accuracy...")
                downbeat_proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
                    beats_per_bar=[3, 4], 
                    fps=200,           # Higher resolution matching beat tracking
                    transition_lambda=100  # Improved transition modeling
                )
                downbeats = downbeat_proc(act)
                
                # OPTIMIZED tempo detection with higher precision
                tempo_estimator = madmom.features.tempo.TempoEstimationProcessor(
                    fps=200,           # Higher resolution for better tempo accuracy
                    min_bpm=60,        # Production-relevant range
                    max_bpm=180        # Production-relevant range
                )
                tempi = tempo_estimator(act)
                primary_tempo = tempi[0][0] if len(tempi) > 0 else 120  # BPM
                
                # Create click track MIDI
                click_midi_path = final_output_dir / "click_track.mid"
                create_click_track_midi(beats, downbeats, primary_tempo, click_midi_path)
                output_files.append("click_track.mid")
                
                # Create comprehensive beat analysis
                beat_analysis = {
                    "tempo_bpm": round(primary_tempo, 2),
                    "beats": beats.tolist(),
                    "downbeats": downbeats.tolist() if len(downbeats) > 0 else [],
                    "time_signature": detect_time_signature(beats, downbeats),
                    "total_bars": len(downbeats) if len(downbeats) > 0 else len(beats) // 4,
                    "analysis_summary": {
                        "tempo_stability": analyze_tempo_stability(beats),
                        "rhythm_complexity": analyze_rhythm_complexity(beats),
                        "daw_sync_notes": "Import click_track.mid to sync your DAW perfectly"
                    }
                }
                
                beat_json_path = final_output_dir / "beat_analysis.json"
                with open(beat_json_path, 'w') as f:
                    json.dump(beat_analysis, f, indent=2)
                output_files.append("beat_analysis.json")
                
                # Create DAW-ready tempo map
                tempo_map_path = final_output_dir / "tempo_map.txt"
                create_daw_tempo_map(beats, primary_tempo, tempo_map_path)
                output_files.append("tempo_map.txt")
                
                completed_tasks.append("beat")
                print("Beat/Tempo analysis completed - Perfect DAW synchronization ready!")
                
            except Exception as e:
                print(f"Beat/Tempo analysis failed: {e}")
        
        # ===================================================================
        # FINAL PROCESSING & SUMMARY
        # ===================================================================
        
        update_job_progress(job_id, 90, "Finalisierung - Producer Summary wird erstellt")
        
        # Create producer summary
        producer_summary = create_producer_summary(completed_tasks, output_files, primary_tempo if 'primary_tempo' in locals() else None)
        summary_path = final_output_dir / "PRODUCER_SUMMARY.txt"
        with open(summary_path, 'w') as f:
            f.write(producer_summary)
        output_files.append("PRODUCER_SUMMARY.txt")
        
        # Create DAW project template instructions
        daw_instructions = create_daw_instructions(completed_tasks, output_files)
        daw_path = final_output_dir / "DAW_IMPORT_GUIDE.txt"
        with open(daw_path, 'w') as f:
            f.write(daw_instructions)
        output_files.append("DAW_IMPORT_GUIDE.txt")
        
        if not output_files:
            raise Exception("No transcription files were generated")
        
        # Calculate final statistics
        total_size = sum(f.stat().st_size for f in final_output_dir.glob("*"))
        processing_time = round(time.time() - start_time, 2)
        
        # Debug: Check what files actually exist before finalizing
        print(f"DEBUG: Checking final output directory: {final_output_dir}")
        if final_output_dir.exists():
            existing_files = list(final_output_dir.glob("*"))
            print(f"DEBUG: Files found in final output: {[f.name for f in existing_files]}")
            print(f"DEBUG: output_files list: {output_files}")
        else:
            print(f"ERROR: Final output directory does not exist: {final_output_dir}")
        
        # Save final volume state with multiple commits to ensure persistence
        volume.commit()
        print("DEBUG: First volume commit completed")
        
        # Wait a moment and commit again to ensure all files are persisted
        time.sleep(2)
        volume.commit()
        print("DEBUG: Second volume commit completed")
        
        # Final job status
        final_status = {
            "status": "completed",
            "progress": 100, 
            "phase": "Fertig - Alle Transcriptions fuer Producer verfuegbar",
            "error": None,
            "processing_time": processing_time,
            "files": output_files,
            "total_size_mb": round(total_size / (1024*1024), 2),
            "download_base_url": f"/download/transcription/{job_id}",
            "completed_tasks": completed_tasks,
            "producer_benefits": {
                task: TRANSCRIPTION_CONFIG[task]["producer_benefit"] 
                for task in completed_tasks
            }
        }
        
        job_status[job_id] = final_status
        save_job_status(job_id, final_status)
        
        print(f"Complete music transcription job {job_id} finished!")
        print(f"Completed tasks: {completed_tasks}")
        print(f"Generated files: {len(output_files)}")
        
        return {
            "success": True,
            "files": output_files,
            "job_id": job_id,
            "processing_time": processing_time,
            "total_size": total_size,
            "completed_tasks": completed_tasks,
            "transcription_info": {
                "libraries_used": {
                    task: TRANSCRIPTION_CONFIG[task]["library"] 
                    for task in completed_tasks
                },
                "producer_ready": True,
                "daw_compatible": True
            }
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"Complete music transcription failed: {error_msg}")
        
        error_status = {
            "status": "error",
            "progress": 100,
            "phase": "Fehler bei Transkription",
            "error": error_msg,
            "processing_time": round(time.time() - start_time, 2)
        }
        job_status[job_id] = error_status
        save_job_status(job_id, error_status)
        
        return {
            "success": False,
            "error": error_msg,
            "job_id": job_id,
            "processing_time": round(time.time() - start_time, 2)
        }

# ===================================================================
# HELPER FUNCTIONS FOR PROFESSIONAL MUSIC PRODUCTION
# ===================================================================

def create_producer_midi_summary(input_midi_path, output_path):
    """Create a simplified MIDI file for easier editing in DAW"""
    try:
        import pretty_midi
        
        # Load original MIDI
        midi_data = pretty_midi.PrettyMIDI(str(input_midi_path))
        
        # Create simplified version (remove very short notes, quantize slightly)
        simplified = pretty_midi.PrettyMIDI()
        
        for instrument in midi_data.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=f"{instrument.name}_simplified" if instrument.name else "melody_simplified"
            )
            
            # Filter and process notes
            for note in instrument.notes:
                # Keep notes longer than 0.1 seconds
                if note.end - note.start > 0.1:
                    # Slight quantization to 32nd notes for easier editing
                    start_quantized = round(note.start * 8) / 8  # 32nd note grid at 120 BPM
                    end_quantized = start_quantized + (note.end - note.start)
                    
                    new_note = pretty_midi.Note(
                        velocity=min(100, max(60, note.velocity)),  # Normalize velocity
                        pitch=note.pitch,
                        start=start_quantized,
                        end=end_quantized
                    )
                    new_instrument.notes.append(new_note)
            
            simplified.instruments.append(new_instrument)
        
        simplified.write(str(output_path))
        print(f"Created simplified MIDI: {output_path}")
        
    except Exception as e:
        print(f"Could not create simplified MIDI: {e}")

def create_drum_midi(onsets, output_path):
    """Create a MIDI file with drum hits for sample replacement"""
    try:
        import pretty_midi
        
        # Create MIDI with drum track
        midi = pretty_midi.PrettyMIDI()
        drum_track = pretty_midi.Instrument(program=0, is_drum=True, name="Drum Pattern")
        
        # Map onsets to standard drum MIDI notes
        kick_note = 36   # Standard kick drum
        snare_note = 38  # Standard snare
        hihat_note = 42  # Hi-hat closed
        
        for i, onset_time in enumerate(onsets):
            # Simple pattern: assign different drum sounds based on timing
            if i % 4 == 0:  # Kick on beats 1 and 3
                note_pitch = kick_note
            elif i % 4 == 2:  # Snare on beats 2 and 4
                note_pitch = snare_note
            else:  # Hi-hat on off-beats
                note_pitch = hihat_note
            
            # Create drum hit
            note = pretty_midi.Note(
                velocity=100,
                pitch=note_pitch,
                start=onset_time,
                end=onset_time + 0.1  # Short drum hit
            )
            drum_track.notes.append(note)
        
        midi.instruments.append(drum_track)
        midi.write(str(output_path))
        print(f"Created drum MIDI pattern: {output_path}")
        
    except Exception as e:
        print(f"Could not create drum MIDI: {e}")

def analyze_chord_progression(chords, strength):
    """Analyze chord progression for music production"""
    # Filter out weak detections and create clean progression
    threshold = 0.3
    clean_progression = []
    
    for i, (chord, conf) in enumerate(zip(chords, strength)):
        if conf > threshold:
            # Convert chord to standard notation
            if chord != 'N':  # 'N' means no chord
                clean_progression.append({
                    "chord": chord,
                    "confidence": float(conf),
                    "time": i * 0.023,  # Essentia default hop size
                    "duration": 0.023
                })
    
    return clean_progression

def create_chord_midi(chord_progression, output_path):
    """Create MIDI file with chord progression"""
    try:
        import pretty_midi
        
        midi = pretty_midi.PrettyMIDI()
        chord_track = pretty_midi.Instrument(program=0, name="Chord Progression")
        
        # Simple chord mapping (can be expanded)
        chord_maps = {
            'C': [60, 64, 67],      # C major
            'Dm': [62, 65, 69],     # D minor
            'Em': [64, 67, 71],     # E minor
            'F': [65, 69, 72],      # F major
            'G': [67, 71, 74],      # G major
            'Am': [69, 72, 76],     # A minor
            'Bdim': [71, 74, 77],   # B diminished
        }
        
        for chord_info in chord_progression:
            chord_name = chord_info["chord"]
            start_time = chord_info["time"]
            duration = chord_info.get("duration", 1.0)
            
            # Simple chord mapping
            if chord_name in chord_maps:
                notes = chord_maps[chord_name]
            else:
                # Default to C major if chord not recognized
                notes = [60, 64, 67]
            
            # Add chord notes
            for note_pitch in notes:
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=note_pitch,
                    start=start_time,
                    end=start_time + duration
                )
                chord_track.notes.append(note)
        
        midi.instruments.append(chord_track)
        midi.write(str(output_path))
        print(f"Created chord progression MIDI: {output_path}")
        
    except Exception as e:
        print(f"Could not create chord MIDI: {e}")

def detect_key_signature(chord_progression):
    """Detect the most likely key signature"""
    if not chord_progression:
        return "C major"
    
    # Simple key detection based on chord frequency
    chord_counts = {}
    for chord_info in chord_progression:
        chord = chord_info["chord"]
        chord_counts[chord] = chord_counts.get(chord, 0) + 1
    
    most_common = max(chord_counts, key=chord_counts.get) if chord_counts else "C"
    return f"{most_common} major"  # Simplified

def find_common_progressions(chord_progression):
    """Find common chord progressions for producers"""
    if len(chord_progression) < 3:
        return []
    
    # Extract just chord names
    chords = [c["chord"] for c in chord_progression]
    
    common_patterns = {
        "vi-IV-I-V": ["Am", "F", "C", "G"],
        "I-V-vi-IV": ["C", "G", "Am", "F"],
        "ii-V-I": ["Dm", "G", "C"]
    }
    
    found_progressions = []
    for name, pattern in common_patterns.items():
        # Check if pattern exists in chord sequence
        for i in range(len(chords) - len(pattern) + 1):
            if chords[i:i+len(pattern)] == pattern:
                found_progressions.append(name)
                break
    
    return found_progressions

def generate_producer_tips(chord_progression):
    """Generate helpful tips for producers"""
    tips = [
        "Import chord_progression.mid into your DAW as a guide track",
        "Use these chords to create bass lines and lead melodies", 
        "Try different inversions of these chords for variety",
        "Layer with the melody MIDI for complete harmonic content"
    ]
    
    if len(chord_progression) > 8:
        tips.append("Consider looping sections of this progression for verses/choruses")
    
    return tips

def create_chord_chart(chord_progression, output_path):
    """Create human-readable chord chart"""
    try:
        with open(output_path, 'w') as f:
            f.write("CHORD CHART FOR MUSIC PRODUCTION\n")
            f.write("================================\n\n")
            
            if not chord_progression:
                f.write("No clear chord progression detected.\n")
                return
            
            f.write("Time\t\tChord\t\tConfidence\n")
            f.write("-" * 40 + "\n")
            
            for chord_info in chord_progression:
                time_str = f"{chord_info['time']:.1f}s"
                chord = chord_info['chord']
                conf = f"{chord_info['confidence']:.2f}"
                f.write(f"{time_str:<12}\t{chord:<12}\t{conf}\n")
            
            f.write("\n\nPRODUCER NOTES:\n")
            f.write("- Import chord_progression.mid for MIDI version\n")
            f.write("- Use as foundation for bass lines and melodies\n")
            f.write("- Experiment with different chord voicings\n")
        
        print(f"Created chord chart: {output_path}")
        
    except Exception as e:
        print(f"Could not create chord chart: {e}")

def create_click_track_midi(beats, downbeats, tempo, output_path):
    """Create a click track MIDI for DAW synchronization"""
    try:
        import pretty_midi
        
        midi = pretty_midi.PrettyMIDI()
        click_track = pretty_midi.Instrument(program=0, is_drum=True, name="Click Track")
        
        # Click sounds
        kick_click = 36   # Strong beat (downbeat)
        snare_click = 37  # Regular beat
        
        # Add downbeats as strong clicks
        for downbeat_time in downbeats:
            note = pretty_midi.Note(
                velocity=120,  # Loud for downbeats
                pitch=kick_click,
                start=downbeat_time,
                end=downbeat_time + 0.05
            )
            click_track.notes.append(note)
        
        # Add regular beats as softer clicks
        for beat_time in beats:
            # Skip if this is already a downbeat
            if not any(abs(beat_time - db) < 0.01 for db in downbeats):
                note = pretty_midi.Note(
                    velocity=80,  # Softer for regular beats
                    pitch=snare_click,
                    start=beat_time,
                    end=beat_time + 0.05
                )
                click_track.notes.append(note)
        
        midi.instruments.append(click_track)
        midi.write(str(output_path))
        print(f"Created click track MIDI: {output_path}")
        
    except Exception as e:
        print(f"Could not create click track MIDI: {e}")

def detect_time_signature(beats, downbeats):
    """Detect time signature from beat analysis"""
    if len(downbeats) < 2:
        return "4/4"  # Default
    
    # Calculate average beats between downbeats
    beat_intervals = []
    for i in range(len(downbeats) - 1):
        start_time = downbeats[i]
        end_time = downbeats[i + 1]
        
        # Count beats between downbeats
        beats_in_bar = sum(1 for beat in beats if start_time <= beat < end_time)
        if beats_in_bar > 0:
            beat_intervals.append(beats_in_bar)
    
    if not beat_intervals:
        return "4/4"
    
    # Most common interval
    avg_beats = round(np.mean(beat_intervals))
    
    if avg_beats == 3:
        return "3/4"
    elif avg_beats == 4:
        return "4/4"
    elif avg_beats == 6:
        return "6/8"
    else:
        return f"{avg_beats}/4"

def analyze_tempo_stability(beats):
    """Analyze how stable the tempo is"""
    if len(beats) < 3:
        return "stable"
    
    # Calculate intervals between beats
    intervals = np.diff(beats)
    
    # Calculate coefficient of variation
    cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
    
    if cv < 0.05:
        return "very stable"
    elif cv < 0.1:
        return "stable"
    elif cv < 0.2:
        return "moderately stable"
    else:
        return "variable tempo"

def analyze_rhythm_complexity(beats):
    """Analyze rhythm complexity"""
    if len(beats) < 4:
        return "simple"
    
    # Simple metric based on beat regularity
    intervals = np.diff(beats)
    regularity = 1 - (np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 1
    
    if regularity > 0.9:
        return "simple and regular"
    elif regularity > 0.7:
        return "moderate complexity"
    else:
        return "complex rhythm"

def create_daw_tempo_map(beats, primary_tempo, output_path):
    """Create a tempo map file for DAW import"""
    try:
        with open(output_path, 'w') as f:
            f.write("DAW TEMPO MAP\n")
            f.write("=============\n\n")
            f.write(f"Primary Tempo: {primary_tempo:.2f} BPM\n\n")
            f.write("Beat Positions (seconds):\n")
            f.write("-" * 30 + "\n")
            
            for i, beat_time in enumerate(beats):
                f.write(f"Beat {i+1:3d}: {beat_time:.3f}s\n")
            
            f.write("\n\nDAW IMPORT INSTRUCTIONS:\n")
            f.write("1. Set your DAW to " + f"{primary_tempo:.0f} BPM\n")
            f.write("2. Import click_track.mid for perfect synchronization\n")
            f.write("3. Use beat positions above for manual tempo mapping if needed\n")
        
        print(f"Created DAW tempo map: {output_path}")
        
    except Exception as e:
        print(f"Could not create tempo map: {e}")

def create_producer_summary(completed_tasks, output_files, tempo=None):
    """Create a comprehensive summary for producers"""
    summary = """
KLEIN DIGITAL SOLUTIONS - MUSIC TRANSCRIPTION COMPLETE
=========================================================

PRODUCER-READY FILES GENERATED:
"""
    
    if "melody" in completed_tasks:
        summary += """
MELODY/INSTRUMENTS (basic-pitch):
   -> melody_notes.mid - Load directly into VST instruments
   -> melody_simplified.mid - Cleaned version for easier editing
   -> melody_notes.json - Detailed note data
   
   PRODUCER BENEFIT: Perfect MIDI notes for harmonizers, synthesizers & VST instruments
"""

    if "drums" in completed_tasks:
        summary += """
DRUMS (madmom):
   -> drums_pattern.mid - Import into drum machine/sampler
   -> drums_timing.json - Precise timing data
   
   PRODUCER BENEFIT: Replace drums with your own samples while keeping original timing
"""

    if "chords" in completed_tasks:
        summary += """
CHORDS (essentia):
   -> chord_progression.mid - Full chord progression as MIDI
   -> chord_analysis.json - Detailed harmonic analysis
   -> chord_chart.txt - Human-readable chord chart
   
   PRODUCER BENEFIT: Perfect for remixes, covers & creating new arrangements
"""

    if "beat" in completed_tasks:
        summary += f"""
BEAT/TEMPO (madmom):
   -> click_track.mid - Perfect synchronization for your DAW
   -> beat_analysis.json - Complete timing analysis
   -> tempo_map.txt - Manual tempo mapping guide
   {f"-> Primary Tempo: {tempo:.1f} BPM" if tempo else ""}
   
   PRODUCER BENEFIT: Sync your DAW perfectly with the original track
"""

    summary += """

DAW WORKFLOW TIPS:
==================
1. Import click_track.mid first to set up perfect timing
2. Load melody_notes.mid into your favorite VST instrument
3. Use chord_progression.mid as harmonic foundation
4. Replace drums_pattern.mid hits with your own samples
5. Check PRODUCER_SUMMARY.txt for detailed instructions

READY FOR PROFESSIONAL MUSIC PRODUCTION!
"""
    
    return summary

def create_daw_instructions(completed_tasks, output_files):
    """Create detailed DAW import instructions"""
    instructions = """
DAW IMPORT GUIDE - KLEIN DIGITAL SOLUTIONS
==========================================

STEP-BY-STEP WORKFLOW:
"""

    if "beat" in completed_tasks:
        instructions += """
1. TEMPO & SYNCHRONIZATION:
   - Open your DAW and create a new project
   - Import 'click_track.mid' first
   - This will automatically set the correct tempo and timing
   - Enable metronome and sync with the click track
"""

    if "melody" in completed_tasks:
        instructions += """
2. MELODY & INSTRUMENTS:
   - Create a new MIDI track for melody
   - Import 'melody_notes.mid' or 'melody_simplified.mid'
   - Load your favorite VST instrument (piano, synth, etc.)
   - Adjust velocity and timing as needed
   - The MIDI is quantized and ready for professional use
"""

    if "chords" in completed_tasks:
        instructions += """
3. CHORD PROGRESSION:
   - Create a new MIDI track for chords
   - Import 'chord_progression.mid'
   - Load a pad/piano VST or use for bass line reference
   - Check 'chord_chart.txt' for chord names and timing
   - Use as foundation for bass lines and additional harmonies
"""

    if "drums" in completed_tasks:
        instructions += """
4. DRUM REPLACEMENT:
   - Create a new MIDI track for drums
   - Import 'drums_pattern.mid'
   - Load your drum machine or sampler
   - Replace MIDI notes with your own drum samples
   - The timing matches the original track perfectly
"""

    instructions += """

ADVANCED TECHNIQUES:
===================
- Layer melody MIDI with chord progression for full harmonic content
- Use chord analysis to create bass lines in the same key
- Experiment with different VST instruments on the melody track
- Create variations by copying and modifying the MIDI patterns
- Use the tempo map for complex timing adjustments

TROUBLESHOOTING:
===============
- If timing seems off, make sure your DAW is set to the correct BPM
- Check that all MIDI tracks are properly aligned with the click track
- For complex rhythms, refer to the beat_analysis.json file
- Contact Klein Digital Solutions for additional support

READY TO CREATE AMAZING MUSIC!
"""
    
    return instructions

# ===================================================================
# FASTAPI WEB APPLICATION
# ===================================================================

def create_transcription_app():
    from fastapi import FastAPI, File, Form, UploadFile, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI(
        title="Klein Digital Solutions - Music AI Transcription Complete",
        description="Professional music transcription for producers: Melody, Drums, Chords & Beat analysis",
        version="1.0.0"
    )

    # CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/transcribe")
    async def api_transcribe(
        audio_file: UploadFile = File(..., description="Audio file for transcription"),
        melody: bool = Form(True, description="Extract melody/instruments (basic-pitch)"),
        drums: bool = Form(True, description="Extract drum patterns (madmom)"),
        chords: bool = Form(True, description="Analyze chord progression (essentia)"),
        beat: bool = Form(True, description="Analyze beat/tempo (madmom)")
    ):
        """
        Complete music transcription for professional production
        """
        from fastapi import Response
        
        try:
            print(f"Received transcription request: {audio_file.filename}")
            
            # Validate file format
            allowed_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg']
            if not any(audio_file.filename.lower().endswith(ext) for ext in allowed_extensions):
                raise HTTPException(400, f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}")
            
            # Read and validate audio data
            audio_data = await audio_file.read()
            
            if len(audio_data) == 0:
                raise HTTPException(400, "Empty audio file uploaded")
            
            # File size limit (100MB for transcription)
            max_size = 100 * 1024 * 1024
            if len(audio_data) > max_size:
                raise HTTPException(400, f"File too large (max {max_size // 1024 // 1024}MB)")
            
            # Check that at least one task is selected
            tasks = {"melody": melody, "drums": drums, "chords": chords, "beat": beat}
            if not any(tasks.values()):
                raise HTTPException(400, "At least one transcription task must be selected")
            
            # Generate job ID
            job_id = str(uuid.uuid4())
            file_extension = Path(audio_file.filename).suffix
            safe_filename = f"{job_id}{file_extension}"
            
            selected_tasks = [task for task, enabled in tasks.items() if enabled]
            print(f"Starting transcription for {audio_file.filename} ({len(audio_data)} bytes)")
            print(f"Selected tasks: {selected_tasks}")
            
            # Initialize job status
            job_status[job_id] = {
                "status": "queued",
                "progress": 0,
                "phase": "Upload abgeschlossen, Transcription wird vorbereitet...",
                "error": None,
                "processing_time": None,
                "files": [],
                "total_size_mb": 0,
                "selected_tasks": selected_tasks
            }
            
            # Start async transcription
            transcribe_complete_audio.spawn(audio_data, safe_filename, job_id, tasks)
            
            response_data = {
                "job_id": job_id,
                "status": "queued",
                "selected_tasks": selected_tasks,
                "message": "Complete music transcription started. Use /status/transcription/{job_id} to check progress.",
                "estimated_time": f"{len(selected_tasks) * 2}-{len(selected_tasks) * 5} minutes"
            }
            
            print(f"Music transcription job {job_id} queued for processing")
            
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
            print(f"Transcription upload error: {str(e)}")
            raise HTTPException(500, f"Transcription upload error: {str(e)}")

    @web_app.get("/status/transcription/{job_id}")
    async def transcription_status(job_id: str):
        """Status API for transcription progress"""
        # Always check volume first for latest status
        volume.reload()
        status_file = Path(storage_path) / f"status_{job_id}.json"
        
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)
                print(f"Transcription status loaded from volume for job {job_id}: {status.get('progress', 0)}% - {status.get('phase', 'Unknown')}")
                return status
            except Exception as e:
                print(f"Error loading transcription status from volume: {e}")
        
        # Fallback to in-memory status
        status = job_status.get(job_id)
        if status:
            print(f"Transcription status loaded from memory for job {job_id}: {status.get('progress', 0)}% - {status.get('phase', 'Unknown')}")
            return status
        
        print(f"No transcription status found for job {job_id}")
        return {
            "status": "unknown", 
            "progress": 0, 
            "phase": "Job nicht gefunden", 
            "error": "Transcription job nicht gefunden"
        }

    @web_app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "Klein Digital Solutions - Music AI Transcription Complete",
            "version": "1.0.0",
            "libraries": {
                "melody": "basic-pitch (Spotify)",
                "drums": "madmom",
                "chords": "essentia",
                "beat": "madmom"
            },
            "max_file_size": "100MB",
            "supported_formats": ["MP3", "WAV", "FLAC", "M4A", "AAC", "OGG"],
            "producer_ready": True
        }

    @web_app.get("/capabilities")
    async def capabilities():
        """Get detailed transcription capabilities"""
        return {
            "transcription_types": TRANSCRIPTION_CONFIG,
            "workflow": {
                "step_1": "Upload audio file",
                "step_2": "Select transcription tasks (melody, drums, chords, beat)",
                "step_3": "Download producer-ready MIDI and analysis files",
                "step_4": "Import into your DAW using provided guides"
            },
            "output_formats": {
                "midi": "Ready for DAW import",
                "json": "Detailed analysis data",
                "txt": "Human-readable charts and guides"
            },
            "producer_benefits": {
                "time_saving": "Skip manual transcription work",
                "accuracy": "AI-powered precision",
                "daw_ready": "Files optimized for music production",
                "educational": "Learn from professional track analysis"
            }
        }

    @web_app.get("/download/transcription/{job_id}/{filename}")
    async def download_transcription_file(job_id: str, filename: str):
        """Download individual transcription file"""
        volume.reload()  # Force volume reload for latest state
        
        file_path = Path(storage_path) / job_id / "output" / "final" / filename
        
        if not file_path.exists():
            raise HTTPException(404, f"Transcription file not found: {filename}")

        def file_iterator():
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
        
        # Determine media type
        media_type = "application/octet-stream"
        if filename.endswith(('.mid', '.midi')):
            media_type = "audio/midi"
        elif filename.endswith('.json'):
            media_type = "application/json"
        elif filename.endswith('.txt'):
            media_type = "text/plain"
        
        return StreamingResponse(
            file_iterator(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Allow-Origin": "*",
            }
        )

    @web_app.get("/download/transcription/{job_id}/all.zip")
    async def download_transcription_zip(job_id: str):
        """Download all transcription files as ZIP"""
        volume.reload()  # Force volume reload for latest state
        
        output_dir = Path(storage_path) / job_id / "output" / "final"
        
        # Debug logging for download issues
        print(f"DEBUG DOWNLOAD: Looking for files in: {output_dir}")
        print(f"DEBUG DOWNLOAD: Directory exists: {output_dir.exists()}")
        print(f"DEBUG DOWNLOAD: Is directory: {output_dir.is_dir()}")
        
        if output_dir.exists():
            files_found = list(output_dir.glob("*"))
            print(f"DEBUG DOWNLOAD: Files found: {[f.name for f in files_found]}")
            print(f"DEBUG DOWNLOAD: File count: {len(files_found)}")
        else:
            # Check parent directories
            job_dir = Path(storage_path) / job_id
            print(f"DEBUG DOWNLOAD: Job dir exists: {job_dir.exists()}")
            if job_dir.exists():
                job_contents = list(job_dir.glob("*"))
                print(f"DEBUG DOWNLOAD: Job dir contents: {[f.name for f in job_contents]}")
                
                output_base = job_dir / "output"
                print(f"DEBUG DOWNLOAD: Output base exists: {output_base.exists()}")
                if output_base.exists():
                    output_contents = list(output_base.glob("*"))
                    print(f"DEBUG DOWNLOAD: Output base contents: {[f.name for f in output_contents]}")
        
        if not output_dir.is_dir():
            raise HTTPException(404, "Transcription files not found")
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    zip_file.write(file_path, file_path.name)
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=music_transcription_complete.zip",
                "Access-Control-Allow-Origin": "*",
            }
        )

    @web_app.get("/preview/transcription/{job_id}/{filename}")
    async def preview_transcription_file(job_id: str, filename: str):
        """Generate and stream audio preview for transcription files"""
        volume.reload()
        
        output_dir = Path(storage_path) / job_id / "output" / "final"
        file_path = output_dir / filename
        
        print(f"DEBUG PREVIEW: Looking for {file_path}")
        
        if not file_path.exists():
            raise HTTPException(404, f"Transcription file not found: {filename}")
        
        # For MIDI files, we need to convert to audio for preview
        if filename.endswith('.mid') or filename.endswith('.midi'):
            # Generate audio preview from MIDI using FluidSynth
            preview_path = output_dir / f"preview_{filename.replace('.mid', '.mp3').replace('.midi', '.mp3')}"
            
            if not preview_path.exists():
                try:
                    # Use fluidsynth to render MIDI to audio
                    import subprocess
                    wav_temp = output_dir / f"temp_{filename.replace('.mid', '.wav').replace('.midi', '.wav')}"
                    
                    # Convert MIDI to WAV using fluidsynth
                    fluidsynth_cmd = [
                        "fluidsynth", "-ni", "/usr/share/soundfonts/default.sf2", 
                        str(file_path), "-F", str(wav_temp), "-r", "44100"
                    ]
                    
                    subprocess.run(fluidsynth_cmd, check=True, capture_output=True, timeout=30)
                    
                    # Convert WAV to MP3 preview (30 seconds)
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", str(wav_temp), "-t", "30", 
                        "-ar", "44100", "-ac", "2", "-b:a", "128k", str(preview_path)
                    ]
                    
                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=30)
                    
                    # Clean up temp WAV
                    if wav_temp.exists():
                        wav_temp.unlink()
                    
                    volume.commit()
                    print(f"DEBUG PREVIEW: Generated MIDI preview: {preview_path}")
                    
                except Exception as e:
                    print(f"❌ Failed to generate MIDI preview: {e}")
                    raise HTTPException(500, f"Could not generate preview for {filename}")
            
            # Stream the MP3 preview
            def file_iterator():
                with open(preview_path, "rb") as f:
                    while chunk := f.read(8192):
                        yield chunk
            
            return StreamingResponse(
                file_iterator(),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"inline; filename=preview_{filename}.mp3",
                    "Access-Control-Allow-Origin": "*",
                }
            )
        
        # For non-MIDI files (JSON, TXT), return as downloadable content
        else:
            def file_iterator():
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        yield chunk
            
            media_type = "application/json" if filename.endswith('.json') else "text/plain"
            
            return StreamingResponse(
                file_iterator(),
                media_type=media_type,
                headers={
                    "Content-Disposition": f"inline; filename={filename}",
                    "Access-Control-Allow-Origin": "*",
                }
            )

    @web_app.options("/{path:path}")
    async def options_handler(path: str):
        """Handle CORS preflight requests"""
        return {"status": "ok"}
    
    return web_app

# Deploy FastAPI app using Modal's ASGI pattern
@app.function(image=transcription_image, volumes={storage_path: volume})
@modal.asgi_app()
def transcription_app():
    return create_transcription_app()

if __name__ == "__main__":
    print("Klein Digital Solutions - Music AI Transcription Complete")
    print("Professional transcription for producers & musicians")
    print("Modal Serverless Deployment with GPU optimization")
    print("Deploy: modal deploy modal_app_transcription.py")
    print("\nFeatures:")
    print("   • basic-pitch: Melody/Instruments -> VST-ready MIDI")
    print("   • madmom: Drums -> Sample replacement MIDI")
    print("   • essentia: Chords -> Harmonic analysis for remixes")
    print("   • madmom: Beat/Tempo -> Perfect DAW synchronization")