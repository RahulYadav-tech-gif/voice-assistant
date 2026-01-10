import pyaudio
import wave
import numpy as np
import threading
import tempfile
import time
from typing import Optional


class AudioRecorder:
    """
    Audio recorder with automatic silence detection.
    Records audio and automatically stops when silence is detected.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        silence_threshold: float = 500,
        silence_duration: float = 2.0
    ):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Audio sample rate in Hz (16000 is optimal for Whisper)
            chunk_size: Number of frames per buffer
            channels: Number of audio channels (1 for mono)
            silence_threshold: RMS energy level below which audio is considered silence
            silence_duration: Seconds of continuous silence before auto-stop
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        
        self.is_recording = False
        self.frames = []
        self.audio = None
        self.stream = None
        self.record_thread = None
        
    def _calculate_rms(self, audio_chunk: bytes) -> float:
        """Calculate RMS (Root Mean Square) energy of audio chunk."""
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2))
        return rms
    
    def _record_audio(self):
        """Internal method to record audio in a separate thread."""
        self.audio = pyaudio.PyAudio()
        
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            silence_start_time = None
            
            while self.is_recording:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    self.frames.append(data)
                    
                    # Calculate RMS energy to detect silence
                    rms = self._calculate_rms(data)
                    
                    if rms < self.silence_threshold:
                        # Silence detected
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif time.time() - silence_start_time >= self.silence_duration:
                            # Silence duration exceeded, stop recording
                            self.is_recording = False
                            break
                    else:
                        # Sound detected, reset silence timer
                        silence_start_time = None
                        
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    break
                    
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
    
    def start_recording(self):
        """Start recording audio."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.frames = []
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.start()
    
    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save to a temporary WAV file.
        
        Returns:
            Path to the saved audio file, or None if no audio was recorded
        """
        # Set flag to false to ensure recording stops
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.record_thread:
            self.record_thread.join(timeout=5.0)
        
        # Save recorded audio to temporary file
        if not self.frames:
            return None
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
        
        return temp_file.name
    
    def is_currently_recording(self) -> bool:
        """Check if currently recording."""
        return self.is_recording
