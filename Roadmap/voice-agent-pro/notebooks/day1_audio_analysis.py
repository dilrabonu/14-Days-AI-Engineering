import librosa
import librosa.display
import matplotlib.pyplot as plt 
import numpy as np 

# load voice audio
file_path = r"C:\Users\user\14-Days-AI-Engineering\Roadmap\voice-agent-pro\data\sample.wav"

y, sr = librosa.load(file_path)

plt.figure(figsize=(12,4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform of Audio Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# plot spectogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(12,5))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f db')
plt.title('Spectrogram')
plt.show()

# Extract MFCC( Mel Frequency Cepstral Coefficients)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=(10,4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title("MFCC")
plt.tight_layout()
plt.show()