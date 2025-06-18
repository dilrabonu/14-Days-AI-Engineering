# 🗣️ AI Logistic Voice Assistant

An intelligent voice assistant for logistics and delivery services — built using **Streamlit**, **ElevenLabs TTS**, and **OpenAI GPT-3.5**.

This assistant can:
- 🔉 Speak logistics messages in natural human-like voices
- 🧠 Automatically generate replies to delivery-related questions using ChatGPT
- 🧾 Log and play back all generated voice messages

---

## 🚀 Features

### 🌍 Language & Voice Selector
Choose from supported languages and realistic voices (English, Spanish, French) using ElevenLabs' multilingual voice models.

### 📦 Smart Templates
Click to insert prewritten logistics messages:
- Delivery confirmation  
- Delay notification  
- Out for delivery  
- Pickup ready  
- Payment reminder  

### 🤖 ChatGPT Integration
Type a question like “Where is my package?”  
The assistant will auto-generate a polite, professional response using OpenAI’s `gpt-3.5-turbo`.

### 🔊 ElevenLabs Voice Synthesis
All messages (manual or AI-generated) are converted to speech using the ElevenLabs API and saved as timestamped `.mp3` files.

### 📁 Audio Log System
Every message is saved and listed in a player-style log with:
- ✅ Timestamped filename  
- ✅ Built-in browser audio playback  
- ✅ Future support for downloading or deletion

---

## 🧑‍💻 Tech Stack

| Tool        | Usage                                      |
|-------------|---------------------------------------------|
| Streamlit   | Web app UI                                  |
| ElevenLabs  | Text-to-Speech (TTS) voice generation       |
| OpenAI      | GPT-3.5-Turbo for message generation        |
| Python      | App logic and API integration               |
| dotenv      | Environment variable management             |

---


🙌 Acknowledgments
ElevenLabs – for ultra-realistic TTS

OpenAI – for GPT-based interaction

Streamlit – for the clean interactive UI

