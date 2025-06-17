import os
import csv
import asyncio
from deepgram import Deepgram
from dotenv import load_dotenv
from jiwer import wer, cer

# ✅ Load .env and Deepgram API key
load_dotenv()
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
dg_client = Deepgram(DEEPGRAM_API_KEY)

# ✅ Define audio + ground truth test cases
test_cases = [
    {'audio': 'clean_audio.wav', 'ground_truth': 'clean_audio_gt.txt'},
    {'audio': 'noisy_audio.wav', 'ground_truth': 'noisy_audio_gt.txt'},
    {'audio': 'heavy_noise.wav', 'ground_truth': 'heavy_noise_gt.txt'},
]

async def benchmark():
    results = []

    for case in test_cases:
        audio_path = case['audio']
        gt_path = case['ground_truth']

        print(f"\n🔎 Processing: {case['audio']}")

        try:
            # ✅ Read ground truth transcript
            with open(gt_path, 'r', encoding='utf-8') as f:
                ground_truth = f.read().strip().lower()

            # ✅ Read and send audio
            with open(audio_path, 'rb') as audio:
                source = {'buffer': audio, 'mimetype': 'audio/wav'}
                response = await dg_client.transcription.prerecorded(source, {'punctuate': True})
                
                print("🔍 Full API Response:", response)

                if 'results' in response:
                    hypothesis = response['results']['channels'][0]['alternatives'][0]['transcript'].lower()
                    wer_score = round(wer(ground_truth, hypothesis), 3)
                    cer_score = round(cer(ground_truth, hypothesis), 3)
                else:
                    print(f"❌ No results returned for {case['audio']}")
                    hypothesis = "TRANSCRIPTION FAILED"
                    wer_score = cer_score = 1.0

        except Exception as e:
            print(f"🚨 Error processing {case['audio']}: {e}")
            hypothesis = "ERROR"
            wer_score = cer_score = 1.0

        results.append({
            'audio_file': case['audio'],
            'WER': wer_score,
            'CER': cer_score,
            'transcript': hypothesis
        })

    # ✅ Save results to CSV
    output_file = "transcription_metrics.csv"
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ All done! Results saved to {output_file}")

# ✅ Run
if __name__ == '__main__':
    asyncio.run(benchmark())

