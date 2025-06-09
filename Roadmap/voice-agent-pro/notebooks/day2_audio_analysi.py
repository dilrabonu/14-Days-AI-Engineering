import librosa
import librosa.display
import matplotlib.pyplot as plt 
import numpy as np 
import scipy

file_path = r"C:\Users\user\14-Days-AI-Engineering\Roadmap\voice-agent-pro\data\day2.wav"
y, sr = librosa.load(file_path)

#Plot Wavefrom
plt.figure(figsize=(10,4))
librosa.display.waveshow(y, sr=sr)
plt.title("Wavefrom of Voice")
plt.xlabel('Time (s)')
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

#Plot Spectrogram
plt.figure(figsize=(10,4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of Voice')
plt.xlabel('Time (s)')
plt.ylabel("Frequency (Hz)")
plt.show()