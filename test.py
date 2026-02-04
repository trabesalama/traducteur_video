import pysrt
from TTS.api import TTS
from pydub import AudioSegment
import os

# -------- CONFIG --------
SRT_FILE = "subtitle_fr.srt"
OUTPUT_AUDIO = "output.wav"
LANG_MODEL = "tts_models/fr/css10/vits"
TMP_DIR = "tmp_audio"
# ------------------------

os.makedirs(TMP_DIR, exist_ok=True)

# Charger le modèle TTS
tts = TTS(model_name=LANG_MODEL, progress_bar=True)

# Charger le SRT
subs = pysrt.open(SRT_FILE)

final_audio = AudioSegment.silent(duration=0)

for i, sub in enumerate(subs):
    text = sub.text.replace("\n", " ")
    start_ms = sub.start.ordinal
    end_ms = sub.end.ordinal
    duration_ms = end_ms - start_ms

    tmp_wav = f"{TMP_DIR}/segment_{i}.wav"

    # Générer la voix
    tts.tts_to_file(text=text, file_path=tmp_wav)

    segment_audio = AudioSegment.from_wav(tmp_wav)

    # Ajuster la durée si nécessaire
    if len(segment_audio) < duration_ms:
        silence = AudioSegment.silent(duration=duration_ms - len(segment_audio))
        segment_audio += silence
    else:
        segment_audio = segment_audio[:duration_ms]

    # Ajouter silence avant si besoin
    if len(final_audio) < start_ms:
        final_audio += AudioSegment.silent(duration=start_ms - len(final_audio))

    final_audio += segment_audio

# Export final
final_audio.export(OUTPUT_AUDIO, format="wav")

print("✅ Audio généré :", OUTPUT_AUDIO)
