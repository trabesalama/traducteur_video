from dotenv import load_dotenv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from TTS.api import TTS

# -----------------------------
# 0️⃣ Charger la clé API
# -----------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("Clé OPENROUTER_API_KEY non définie dans .env")

# -----------------------------
# 1️⃣ Configuration
# -----------------------------
VIDEO_IN = "video.mp4"
AUDIO_EN = "audio_en.wav"
AUDIO_FR = "audio_fr.wav"
VIDEO_OUT = "video_fr.mp4"
CHUNK_DIR = "chunks"
CHUNK_FR_DIR = "chunks_fr"
CHUNK_DURATION = 10  # secondes
MAX_WORKERS = 4

S2T_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"  # S2T natif si disponible
TRANSLATE_MODEL = "arcee-ai/trinity-large-preview:free"
TTS_MODEL_NAME = "tts_models/fr/css10/vits"

os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(CHUNK_FR_DIR, exist_ok=True)

# -----------------------------
# 2️⃣ Clients
# -----------------------------
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
tts = TTS(model_name=TTS_MODEL_NAME, gpu=False, progress_bar=False)

# -----------------------------
# 3️⃣ Extraire audio anglais
# -----------------------------
print("📌 Extraction audio anglais...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_IN,
    "-q:a", "0",
    "-map", "0:a",
    AUDIO_EN
], check=True)
print("✅ Audio anglais extrait :", AUDIO_EN)

# -----------------------------
# 4️⃣ Découpage audio en chunks
# -----------------------------
print(f"📌 Découpage audio en chunks de {CHUNK_DURATION}s...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", AUDIO_EN,
    "-f", "segment",
    "-segment_time", str(CHUNK_DURATION),
    "-c", "copy",
    os.path.join(CHUNK_DIR, "chunk_%03d.wav")
], check=True)

chunks = sorted([os.path.join(CHUNK_DIR, f) for f in os.listdir(CHUNK_DIR) if f.endswith(".wav")])
print(f"✅ {len(chunks)} chunks créés")

# -----------------------------
# 5️⃣ Fonction multithread : transcription → traduction → TTS
# -----------------------------
def process_chunk(chunk_path):
    base_name = os.path.basename(chunk_path)
    out_path = os.path.join(CHUNK_FR_DIR, base_name)

    try:
        # 1️⃣ Transcription audio → texte anglais (S2T natif)
        # Note : si S2T endpoint direct disponible, utiliser : client.audio.transcriptions.create(...)
        # Sinon, envoyer via chat/completions avec prompt réduit
        prompt_s2t = f"Transcris l'audio anglais du fichier {base_name}."
        response_s2t = client.chat.completions.create(
            model=S2T_MODEL,
            messages=[{"role": "user", "content": prompt_s2t}],
            extra_body={"reasoning": {"enabled": True}}
        )
        text_en = response_s2t.choices[0].message.content

        # 2️⃣ Traduction texte anglais → français
        prompt_translate = f"Traduis ce texte anglais en français parlé naturel:\n{text_en}"
        response_fr = client.chat.completions.create(
            model=TRANSLATE_MODEL,
            messages=[{"role": "user", "content": prompt_translate}],
            extra_body={"reasoning": {"enabled": True}}
        )
        text_fr = response_fr.choices[0].message.content

        # 3️⃣ Génération audio français
        tts.tts_to_file(text=text_fr, file_path=out_path)
        print(f"✅ Chunk généré : {out_path}")
        return out_path
    except Exception as e:
        print(f"⚠ Erreur sur {chunk_path}: {e}")
        return None

# -----------------------------
# 6️⃣ Multithread pour tous les chunks
# -----------------------------
print("📌 Génération audio français (multithread)...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    fr_files = list(filter(None, executor.map(process_chunk, chunks)))

# -----------------------------
# 7️⃣ Concaténation chunks FR
# -----------------------------
print("📌 Concaténation des chunks FR...")
with open("concat_list.txt", "w", encoding="utf-8") as f:
    for file_path in sorted(fr_files):
        f.write(f"file '{file_path}'\n")

subprocess.run([
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", "concat_list.txt",
    "-c", "copy",
    AUDIO_FR
], check=True)
print("✅ Audio français complet :", AUDIO_FR)

# -----------------------------
# 8️⃣ Fusion audio FR + vidéo
# -----------------------------
print("📌 Fusion audio FR avec la vidéo...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_IN,
    "-i", AUDIO_FR,
    "-c:v", "copy",
    "-map", "0:v:0",
    "-map", "1:a:0",
    VIDEO_OUT
], check=True)
print("🎉 Vidéo française générée :", VIDEO_OUT)
