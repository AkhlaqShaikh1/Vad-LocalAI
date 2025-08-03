import json
import base64
import numpy as np
import logging
import asyncio
from typing import Dict, Any, Optional
from collections import deque
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from scipy.signal import butter, filtfilt

from faster_whisper import WhisperModel


logger = logging.getLogger("LocalAIService.WebSocket")
logger.setLevel(logging.INFO)


router = APIRouter()

active_connections: Dict[str, WebSocket] = {}
vad_processors: Dict[str, 'VADProcessor'] = {}  # Per-channel VAD processors
model = None

class VADProcessor:
    """Voice Activity Detection and Audio Processing"""
    
    def __init__(self, sample_rate=16000, websocket=None, channel_id=None):
        self.sample_rate = sample_rate
        self.websocket = websocket
        self.channel_id = channel_id
        
        # VAD parameters
        self.energy_threshold = 0.003  # Energy threshold for speech detection
        self.zero_crossing_threshold = 0.15  # Zero crossing rate threshold
        self.speech_frames_required = 3  # Consecutive frames needed to detect speech
        self.silence_frames_required = 8  # Consecutive silence frames to end speech
        
        # State tracking
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.is_speech_active = False
        self.last_speech_time = 0
        
        # Audio buffering
        self.audio_buffer = []  # Buffer for accumulating speech audio
        self.frame_buffer = deque(maxlen=5)  # Small buffer for smoothing decisions
        
        # Noise reduction
        self.noise_profile = None
        self.noise_samples = []
        self.noise_collection_complete = False
        self.frames_processed = 0
        
        # High-pass filter for noise reduction (removes low-frequency noise)
        nyquist = sample_rate / 2
        low_cutoff = 80  # Remove frequencies below 80Hz
        self.hp_b, self.hp_a = butter(4, low_cutoff / nyquist, btype='high')
        
        logger.info(f"VAD Processor initialized for channel {channel_id}")
    
    def apply_noise_reduction(self, audio_data):
        """Apply basic noise reduction techniques"""
        try:
            # Apply high-pass filter to remove low-frequency noise
            filtered_audio = filtfilt(self.hp_b, self.hp_a, audio_data)
            
            # Simple spectral subtraction if we have a noise profile
            if self.noise_profile is not None and len(self.noise_profile) > 0:
                # Very basic noise reduction - subtract average noise level
                noise_level = np.mean(np.abs(self.noise_profile))
                reduced_audio = np.where(np.abs(filtered_audio) > noise_level * 1.5, 
                                       filtered_audio, 
                                       filtered_audio * 0.3)
                return reduced_audio
            
            return filtered_audio
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data
    
    def calculate_features(self, audio_frame):
        """Calculate audio features for VAD"""
        if len(audio_frame) == 0:
            return 0, 0
        
        # Energy calculation
        energy = np.sum(audio_frame ** 2) / len(audio_frame)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_frame))))
        zcr = zero_crossings / (2 * len(audio_frame))
        
        return energy, zcr
    
    def is_speech_frame(self, audio_frame):
        """Determine if current frame contains speech"""
        energy, zcr = self.calculate_features(audio_frame)
        
        # Collect noise profile from first few frames (assuming initial silence)
        if not self.noise_collection_complete and self.frames_processed < 20:
            self.noise_samples.extend(audio_frame.tolist())
            if self.frames_processed == 19:  # After 20 frames
                self.noise_profile = np.array(self.noise_samples)
                self.noise_collection_complete = True
                # Update energy threshold based on noise level
                noise_energy = np.sum(self.noise_profile ** 2) / len(self.noise_profile)
                self.energy_threshold = max(noise_energy * 3, 0.003)
                logger.info(f"Noise profile collected. Energy threshold: {self.energy_threshold:.6f}")
        
        self.frames_processed += 1
        
        # Speech detection logic
        is_speech = (energy > self.energy_threshold and 
                    zcr > 0.02 and zcr < 0.8)  # Reasonable ZCR range for speech
        
        # Add to frame buffer for smoothing
        self.frame_buffer.append(is_speech)
        
        # Use majority vote from recent frames
        recent_speech_votes = sum(self.frame_buffer)
        smoothed_is_speech = recent_speech_votes >= len(self.frame_buffer) // 2
        
        return smoothed_is_speech
    
    async def process_audio_chunk(self, audio_bytes):
        """Process incoming audio chunk and handle VAD logic"""
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Apply noise reduction
            if len(audio_np) > 0:
                audio_np = self.apply_noise_reduction(audio_np)
            
            # Check if this frame contains speech
            contains_speech = self.is_speech_frame(audio_np)
            
            current_time = time.time()
            
            if contains_speech:
                self.speech_frames_count += 1
                self.silence_frames_count = 0
                
                # Add audio to buffer
                self.audio_buffer.extend(audio_np)
                
                # Start speech detection
                if not self.is_speech_active and self.speech_frames_count >= self.speech_frames_required:
                    self.is_speech_active = True
                    await self.send_status("🗣️ Speech detected - Recording...")
                    logger.info(f"Speech started for channel {self.channel_id}")
                
                self.last_speech_time = current_time
                
            else:
                self.silence_frames_count += 1
                self.speech_frames_count = max(0, self.speech_frames_count - 1)
                
                # If we were detecting speech, continue buffering for a bit
                if self.is_speech_active:
                    self.audio_buffer.extend(audio_np)  # Include some silence
                
                # End speech detection
                if (self.is_speech_active and 
                    self.silence_frames_count >= self.silence_frames_required and
                    len(self.audio_buffer) > 0):
                    
                    await self.process_speech_segment()
                    self.reset_speech_state()
            
            # Timeout check - if speech was active but no activity for too long
            if (self.is_speech_active and 
                current_time - self.last_speech_time > 3.0 and  # 3 seconds timeout
                len(self.audio_buffer) > 0):
                
                await self.process_speech_segment()
                self.reset_speech_state()
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
    
    async def process_speech_segment(self):
        """Process the accumulated speech segment"""
        if len(self.audio_buffer) == 0:
            return
        
        try:
            await self.send_status("Processing speech...")
            
            # Convert buffer to numpy array
            speech_audio = np.array(self.audio_buffer, dtype=np.float32)
            
            # Only process if we have enough audio (minimum 0.5 seconds)
            min_samples = int(0.5 * self.sample_rate)
            if len(speech_audio) < min_samples:
                logger.info(f"Speech segment too short ({len(speech_audio)} samples), skipping")
                await self.send_status("Speech too short - Skipped")
                return
            
            logger.info(f"Processing speech segment: {len(speech_audio)} samples ({len(speech_audio)/self.sample_rate:.2f}s)")
            
            # Transcribe the speech segment
            transcription = await transcribe_audio_segment(speech_audio)
            
            if transcription and transcription.strip() and "Error" not in transcription:
                await self.send_transcription(transcription.strip())
                await self.send_status("Ready - Listening...")
            else:
                await self.send_status("No speech detected - Listening...")
                
        except Exception as e:
            logger.error(f"Error processing speech segment: {e}", exc_info=True)
            await self.send_status(f"Error: {str(e)}")
    
    def reset_speech_state(self):
        """Reset speech detection state"""
        self.is_speech_active = False
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.audio_buffer = []
    
    async def send_status(self, status_message):
        """Send status update to client"""
        try:
            if self.websocket:
                await self.websocket.send_text(f"Status: {status_message}")
        except Exception as e:
            logger.warning(f"Failed to send status: {e}")
    
    async def send_transcription(self, transcription):
        """Send transcription result to client"""
        try:
            if self.websocket:
                await self.websocket.send_text(f"Transcription: {transcription}")
        except Exception as e:
            logger.warning(f"Failed to send transcription: {e}")

def initialize_whisper_model():
    """Initialize the whisper model once on startup"""
    global model
    try:
        logger.info("Loading faster-whisper model...")
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        logger.info("faster-whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading whisper model: {e}")
        model = None

# Initialize the model when module loads
initialize_whisper_model()

async def transcribe_audio_segment(audio_np):
    """Process audio segment with faster-whisper"""
    global model
    if model is None:
        initialize_whisper_model()
        if model is None:
            logger.error("Whisper model is not initialized. Cannot transcribe audio.")
            return "Error: Could not initialize whisper model"
    
    try:
        logger.info(f"Transcribing audio segment: {len(audio_np)} samples")
        
        # Perform transcription with VAD enabled in Whisper
        segments, info = model.transcribe(
            audio_np, 
            beam_size=10,
            vad_filter=True,  # Enable Whisper's built-in VAD
            vad_parameters=dict(min_silence_duration_ms=200)
        )
        
        # Extract text from segments
        result = " ".join([segment.text.strip() for segment in segments if segment.text.strip()])
        
        # Log transcription stats
        logger.info(f"Transcription complete: '{result}' (detected language: {info.language})")
        return result
        
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return f"Error during transcription: {str(e)}"

# --- WebSocket Endpoint ---
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, channel_id: str):
    vad_processor = None
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket connected for channel: {channel_id}")
        active_connections[channel_id] = websocket
        
        # Create VAD processor for this connection
        vad_processor = VADProcessor(
            sample_rate=16000, 
            websocket=websocket, 
            channel_id=channel_id
        )
        vad_processors[channel_id] = vad_processor
        
        await websocket.send_text("Status: 🎤 Connected - Collecting noise profile...")
        connection_active = True

        while connection_active:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=5.0)
                
                if "text" in message:
                    text_data = message["text"]
                    
                    try:
                        json_data = json.loads(text_data)
                        
                        if "type" in json_data and json_data["type"] == "input_audio_buffer.append" and "audio" in json_data:
                            # Decode base64 audio
                            audio_base64 = json_data["audio"]
                            audio_bytes = base64.b64decode(audio_base64)
                            
                            # Process with VAD
                            await vad_processor.process_audio_chunk(audio_bytes)
                            
                        else:
                            await websocket.send_text("Unknown message type")
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON")
                        await websocket.send_text("Error: Invalid JSON message")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}", exc_info=True)
                        await websocket.send_text(f"Error: {str(e)}")
                
                else:
                    key_str = ", ".join(message.keys()) if hasattr(message, 'keys') else str(message)
                    logger.warning(f"Received message in unknown format. Keys: {key_str}")
            
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for channel: {channel_id}")
                connection_active = False
            except RuntimeError as e:
                if "WebSocket is disconnected" in str(e) or "already completed" in str(e):
                    logger.info(f"WebSocket disconnected for channel: {channel_id}")
                    connection_active = False
                else:
                    logger.error(f"Runtime error: {e}")
                    connection_active = False
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                connection_active = False

    except Exception as e:
        logger.error(f"WebSocket error for channel {channel_id}: {e}", exc_info=True)
    finally:
        # Cleanup
        if channel_id in active_connections:
            del active_connections[channel_id]
        if channel_id in vad_processors:
            del vad_processors[channel_id]
        logger.info(f"Cleaned up connection for channel: {channel_id}")