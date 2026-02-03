import os
import pysrt
import subprocess
import requests
from pydub import AudioSegment
from TTS.api import TTS
from dotenv import load_dotenv
load_dotenv(override=True)
# -----------------------------
# CONFIG VIDEO / SRT
# -----------------------------
VIDEO_IN = "video.mp4"
SRT_FILE = "subtitle.srt"
AUDIO_OUT = "dub_fr.wav"
VIDEO_OUT = "output_dubbed.mp4"

# -----------------------------
# SITES + MODELES DISPONIBLES
# -----------------------------
SITES = [
    # {
    #     "name": "Groq",
    #     "api_url": "https://api.groq.com/openai/v1/chat/completions",
    #     "api_key_env": "GROQ_API_KEY",
    #     "models": [
    #         "llama-3.1-8b-instant",
    #         "llama-3.0-7b-instant"
    #     ]
    # },
    {
        "name": "OpenRouter",
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
        "models": [
            "mistralai/mistral-7b-instruct",
            "meta-llama/llama-3-8b-instruct"
        ]
    }
]

# -----------------------------
# INIT TTS COQUI
# -----------------------------
tts = TTS(
    model_name="tts_models/fr/css10/vits",
    progress_bar=False,
    gpu=False
)

# -----------------------------
# FONCTION DE TRADUCTION SEGMENT
# -----------------------------
def translate_segment(text, sites=SITES, site_idx=0, model_idx=0):
    if site_idx >= len(sites):
        raise RuntimeError("Tous les sites et modèles sont épuisés.")

    site = sites[site_idx]
    api_key = os.getenv(site["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Clé API non définie pour {site['name']}")

    model = site["models"][model_idx]

    try:
        response = requests.post(
            site["api_url"],
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Translate from English to natural spoken French. Do not explain."},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.2
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # Limite de token ou timeout → passer au modèle suivant
        print(f"⚠ Erreur modèle {model} sur site {site['name']}: {e}")
        next_model_idx = model_idx + 1
        next_site_idx = site_idx
        if next_model_idx >= len(site["models"]):
            next_model_idx = 0
            next_site_idx += 1
        return translate_segment(text, sites, site_idx=next_site_idx, model_idx=next_model_idx)

# -----------------------------
# LECTURE SRT
# -----------------------------
subs = pysrt.open(SRT_FILE)
final_audio = AudioSegment.silent(duration=0)

# -----------------------------
# GÉNÉRATION AUDIO PAR SOUS-TITRE
# -----------------------------
for sub in subs:
    text_en = sub.text.replace("\n", " ")
    print(f"▶ Traduction : {text_en[:60]}...")

    # 🔹 Traduction avec multi-modèle/multi-site
    text_fr = translate_segment(text_en)
    print(f"   → FR : {text_fr[:60]}...")

    # Durée du sous-titre
    duration_ms = (
        (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
    ) - (
        (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
    )

    temp_wav = "temp.wav"
    tts.tts_to_file(text=text_fr, file_path=temp_wav)
    speech = AudioSegment.from_wav(temp_wav)

    if len(speech) > duration_ms:
        speech = speech[:duration_ms]
    else:
        speech += AudioSegment.silent(duration=duration_ms - len(speech))

    final_audio += speech
    os.remove(temp_wav)

# -----------------------------
# EXPORT AUDIO
# -----------------------------
final_audio.export(AUDIO_OUT, format="wav")

# -----------------------------
# FUSION AUDIO / VIDÉO
# -----------------------------
subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_IN,
    "-i", AUDIO_OUT,
    "-map", "0:v",
    "-map", "1:a",
    "-c:v", "copy",
    VIDEO_OUT
], check=True)

print("✅ Vidéo doublée générée :", VIDEO_OUT)
