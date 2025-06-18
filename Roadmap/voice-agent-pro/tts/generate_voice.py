import streamlit as st
from elevenlabs import generate, save, set_api_key
from dotenv import load_dotenv
import openai
import os
from datetime import datetime
from openai import OpenAI

# Load environment variables
load_dotenv()
set_api_key(os.getenv("ELEVEN_API_KEY"))

client = OpenAI() 

# Ensure logs directory exists
AUDIO_DIR = "tts/audio_logs"
os.makedirs(AUDIO_DIR, exist_ok=True)

#  Language and voice options
language_voice_map = {
    "English": ["Chris", "Lily", "Daniel"],
    "Spanish": ["Eric", "Jessica"],
    "French": ["Alice", "Will"]
}

#  Smart logistics templates
logistics_templates = {
    "📦 Delivery Confirmation": "Good news! Your package has been successfully delivered.",
    "🕒 Delay Notification": "We’re sorry! Due to unexpected weather conditions, your shipment has been delayed and will arrive tomorrow.",
    "🚚 Out for Delivery": "Your package is on the way and will be delivered by 6 PM today.",
    "✅ Pickup Ready": "Your parcel is ready for pickup at your nearest service center.",
    "💳 Payment Reminder": "A reminder: your payment is pending. Please complete it to receive your package."
}

# -------------- STREAMLIT UI ------------------
st.set_page_config(page_title="AI Logistic Voice Assistant", layout="centered")
st.title("🗣️ Ai Logistic Voice Assistant")

# Step 1: Language & Voice Selection
st.subheader("🌍 Select Language and Voice")
selected_language = st.selectbox("Choose Language:", list(language_voice_map.keys()))
selected_voice = st.selectbox("Choose Voice:", language_voice_map[selected_language])

# Step 2: Smart Templates
st.subheader("📦 Choose a Template (Optional)")
text = ""
cols = st.columns(len(logistics_templates))
for i, (label, template) in enumerate(logistics_templates.items()):
    if cols[i].button(label):
        text = template

# Step 3: Text Area Input
text = st.text_area("✏️ Enter logistic message:", value=text)

# Step 4: Use ChatGPT to Auto-generate Response
use_ai = st.checkbox("🧠 Auto-generate reply using ChatGPT")
if use_ai:
    user_query = st.text_input("What would you like to ask the logistics assistant?", "Where is my package?")
    if st.button("Ask AI"):
        with st.spinner("Generating reply from AI..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a polite logistics assistant. Always reply professionally to customers asking about shipping, delivery, or pickup."},
                        {"role": "user", "content": user_query}
                    ]
                )
                text = response.choices[0].message.content
                st.success("✅ AI Response:")
                st.write(text)
            except Exception as e:
                st.error(f"❌ Error from OpenAI: {e}")

# Step 5: Generate Voice with ElevenLabs
if st.button("🔊 Generate Voice"):
    try:
        audio = generate(
            text=text,
            voice=selected_voice,
            model="eleven_multilingual_v1"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{AUDIO_DIR}/tts_output_{timestamp}.mp3"
        save(audio, output_path)
        st.success("✅ Voice generated successfully!")
        st.audio(output_path, format="audio/mp3")
    except Exception as e:
        st.error(f"❌ Error generating voice: {e}")

# Step 6: Display Previous Logs
st.subheader("📁 Playback Previous Audio Logs")
audio_files = sorted(os.listdir(AUDIO_DIR), reverse=True)

for file in audio_files:
    file_path = os.path.join(AUDIO_DIR, file)
    st.markdown(f"**{file}**")
    st.audio(file_path, format="audio/mp3")
