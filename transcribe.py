import os
import subprocess
import whisper
from deep_translator import GoogleTranslator
from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS, GUJARATI, KANNADA, TELUGU
from dotenv import load_dotenv
import openai

# Load API Keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# File Paths
video_path = "video.mp4"
audio_path = "audio.wav"

# Language Map for Native Transliteration
lang_map = {
    "hindi": DEVANAGARI,
    "gujarati": GUJARATI,
    "kannada": KANNADA,
    "telugu": TELUGU
}

print("Choose Target Language for Captions:")
print("Options → english / hindi / gujarati / kannada / telugu / hinglish")
target_lang = input("Enter language: ").lower()

# Step 1: Extract Audio from Video
print("\nExtracting audio from video...")
subprocess.call(['ffmpeg', '-loglevel', 'quiet', '-i', video_path, '-ab', '160k', '-ac', '2', '-ar', '44100', '-vn', audio_path])
print("Audio extracted successfully!\n")

# Step 2: Load Whisper Model
print("Loading Whisper model...")
model = whisper.load_model("base")

# Step 3: Transcribe Audio to English Text
print("Transcribing audio...")
result = model.transcribe(audio_path)
english_text = result["text"]
print("\nEnglish Transcription:\n", english_text)

# Step 4: Translation + Transliteration based on Target Language
print("\nProcessing Translation...")
if target_lang == 'english':
    final_text = english_text

elif target_lang == 'hinglish':
    print("Translating to Hindi...")
    translated_text = GoogleTranslator(source='auto', target='hi').translate(english_text)
    print("Hindi Translation:\n", translated_text)

    print("Generating Hinglish Caption...")
    final_text = transliterate(translated_text, DEVANAGARI, ITRANS)

else:
    print(f"Translating to {target_lang.title()}...")
    translated_text = GoogleTranslator(source='auto', target=target_lang).translate(english_text)
    print("Translation:\n", translated_text)

    if target_lang in lang_map:
        print("Generating Native Script Caption...")
        final_text = transliterate(translated_text, ITRANS, lang_map[target_lang])
    else:
        final_text = translated_text

print("\nFinal Captions Output:\n", final_text)
