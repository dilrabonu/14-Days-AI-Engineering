import os
folders = ['vad', 'stt', 'tts', 'api', 'ui', 'data', 'notebooks']

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, '.gitkeep'), 'w') as f:
        f.write("")