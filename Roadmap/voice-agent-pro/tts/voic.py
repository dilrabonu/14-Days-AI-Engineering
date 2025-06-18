from elevenlabs import voices, set_api_key
import os
from dotenv import load_dotenv

load_dotenv()
set_api_key(os.getenv("ELEVEN_API_KEY"))

print("✅ Available voices:")
for v in voices():
    print(f"- {v.name}")
