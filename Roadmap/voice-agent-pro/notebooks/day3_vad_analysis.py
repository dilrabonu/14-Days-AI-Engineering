import torch 
import os
from pydub import AudioSegment
from IPython.display import Audio

# ✅ Step 1: Convert to mono + 16kHz for Windows compatibility
input_path = r"C:\Users\user\14-Days-AI-Engineering\Roadmap\voice-agent-pro\data\day3.wav"
converted_path = "day3_converted.wav"

# Convert using pydub (safe even without ffmpeg)
audio = AudioSegment.from_wav(input_path)
audio = audio.set_frame_rate(16000).set_channels(1)
audio.export(converted_path, format="wav")

# ✅ Step 2: Load Silero VAD
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# ✅ Step 3: Load the converted .wav using Silero’s read_audio (returns tensor)
wav = read_audio(converted_path, sampling_rate=16000)

# ✅ Step 4: Get timestamps
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
print("Speech Segments (timestamps):")
print(speech_timestamps)

# ✅ Step 5: Extract only speech parts
cleaned_audio = collect_chunks(wav, speech_timestamps)

# ✅ Step 6: Save output
output_path = "day3_cleaned.wav"
save_audio(output_path, cleaned_audio, sampling_rate=16000)
print(f"✅ Cleaned audio saved to: {output_path}")
