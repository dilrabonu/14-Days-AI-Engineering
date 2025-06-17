import os
from deepgram import Deepgram
import asyncio
from dotenv import load_dotenv

load_dotenv()
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')

dg_client = Deepgram(DEEPGRAM_API_KEY)

async def transcribe_audio(file_path):
    with open(file_path, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': 'audio/wav'}
        response = await dg_client.transcription.prerecorded(source, {'punctuate': True})
        
        print("🔍 Raw Response:\n", response)  # Debug line

        if 'results' in response:
            transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
            print('\n📝 TRANSCRIPT:')
            print(transcript)
        else:
            print("❗Transcription failed. Check the error above.")

        output_path = os.path.join(os.path.dirname(file_path), "transcript.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"\n Transcript saved to: {output_path}")

if __name__ == "__main__":
    audio_path = 'sample_audio.wav'
    asyncio.run(transcribe_audio(audio_path))
