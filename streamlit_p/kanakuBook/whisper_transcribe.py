import sounddevice as sd
import numpy as np
import whisper
import scipy.io.wavfile as wav
import tempfile

DURATION = 5  # seconds
SAMPLE_RATE = 16000  # Hz (Whisper prefers 16000)

print("🎙️ Recording will start in 2 seconds...")
sd.sleep(2000)
print(f"🎤 Recording for {DURATION} seconds...")

# Record audio
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()

print("✅ Recording complete.")

# Save to a temporary .wav file
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
    wav.write(temp_audio_file.name, SAMPLE_RATE, recording)
    audio_path = temp_audio_file.name

# Load Whisper model
print("🧠 Loading Whisper model...")
model = whisper.load_model("base")  # or "small", "medium", "large"

# Transcribe
print("📝 Transcribing...")
result = model.transcribe(audio_path)

print("\n🗒️ Transcription Result:")
print(result["text"])
