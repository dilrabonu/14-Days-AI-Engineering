import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv('transcription_metrics.csv')

df = df[df['transcript'] != 'ERROR']

plt.figure(figsize=(10,6))
plt.bar(df['audio_file'], df['WER'], color='orange', label='WER')
plt.bar(df['audio_file'], df['CER'], color='blue', label='CER', alpha=0.7)
plt.title('Speech-to-Text Performance (WER & CER)')
plt.ylabel('Error Rate')
plt.legend()
plt.ylim(0, 1)
plt.grid(True)
plt.show()