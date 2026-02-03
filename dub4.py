import os
import asyncio
import argparse
import json
import time
import threading
# tqdm optional
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

from dotenv import load_dotenv
from groq import Groq
import edge_tts
import base64
import io
import hashlib
import shutil
from pathlib import Path
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip

# 1. Chargement de la clé API depuis le fichier .env
load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("La clé API GROQ n'est pas définie dans le fichier .env")

# Liste de modèles préférés (séparés par des virgules) - modifiable via variable d'environnement GROQ_MODELS
# Exemple : export GROQ_MODELS="meta-llama/llama-4-scout-17b-16e-instruct,llama-3.3-70b-versatile"
GROQ_MODEL_PREFERENCES = [m.strip() for m in os.getenv("GROQ_MODELS",
 "meta-llama/llama-4-scout-17b-16e-instruct," \
 "llama-3.3-70b-versatile,llama-3.1-8b-instant," \
 "canopylabs/orpheus-arabic-saudi," \
 "canopylabs/orpheus-v1-english," \
 "meta-llama/llama-4-maverick-17b-128e-instruct," \
 "meta-llama/llama-4-scout-17b-16e-instruct").split(",") if m.strip()]

# Maximum de caractères (approx) accepté par Groq avant d'utiliser Google pour éviter les erreurs de tokens
GROQ_MAX_CHARS = int(os.getenv("GROQ_MAX_CHARS", "2000"))

# Translation cache configuration
TRANSLATION_CACHE_PATH = os.getenv("TRANSLATION_CACHE_PATH", os.path.join('.', "translation_cache.json"))
TRANSLATION_CACHE_ENABLED = True
TRANSLATION_CACHE_TTL = int(os.getenv("TRANSLATION_CACHE_TTL", str(7 * 24 * 3600)))  # seconds
_translation_cache_lock = threading.Lock()
_translation_cache = None
# runtime counters
TRANSLATION_CACHE_HITS = 0
TRANSLATION_CACHE_MISSES = 0

# Audio cache configuration (text -> mp3)
AUDIO_CACHE_ENABLED = True
AUDIO_CACHE_PATH = os.getenv("AUDIO_CACHE_PATH", os.path.join('.', "audio_cache"))
AUDIO_CACHE_TTL = int(os.getenv("AUDIO_CACHE_TTL", str(7 * 24 * 3600)))  # seconds
_audio_cache_locks: dict = {}
# runtime counters
AUDIO_CACHE_HITS = 0
AUDIO_CACHE_MISSES = 0


# 2. Configuration des fichiers
INPUT_VIDEO = "video.mp4"  # Vidéo dans le répertoire courant
OUTPUT_VIDEO = "video_output_french_synced.mp4"
TEMP_AUDIO_EN = "temp_audio_en.wav"
TEXT_OUTPUT = "transcript.txt"
FRENCH_TEXT = "french_translation.txt"
TEMP_DIR = "temp_tts_segments"

# --- UTILITAIRES SRT ---
import re
_time_re = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")

def _ts_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    blocks = re.split(r"\n\s*\n", content)
    subs = []
    for b in blocks:
        lines = [l for l in b.splitlines() if l.strip()]
        if not lines:
            continue
        time_line = None
        for line in lines:
            if _time_re.search(line):
                time_line = line
                break
        if not time_line:
            continue
        m = _time_re.search(time_line)
        start = _ts_to_seconds(m.group(1))
        end = _ts_to_seconds(m.group(2))
        idx = lines.index(time_line)
        text = "\n".join(lines[idx + 1 :])
        subs.append({"start": start, "end": end, "text": text})
    return subs

# --- NETTOYAGE / UTIL ---
BASE_WPS = 3.0
MIN_RATE = 50
MAX_RATE = 400

def compute_rate_percent(text: str, target_duration: float, base_wps: float = BASE_WPS):
    words = len(text.split())
    if words == 0 or target_duration <= 0.01:
        return 100
    est = words / base_wps
    percent = int((est / target_duration) * 100)
    percent = max(MIN_RATE, min(MAX_RATE, percent))
    return percent


def shorten_text_to_fit(text: str, target_duration: float) -> str:
    words = text.split()
    if not words:
        return text
    allowed_words = int(target_duration * BASE_WPS * (MAX_RATE / 100.0))
    if allowed_words < 1:
        allowed_words = 1
    if len(words) <= allowed_words:
        return text
    shortened = ' '.join(words[:allowed_words])
    if len(shortened) < len(text):
        shortened = shortened.rstrip('.,;:!?') + '...'
    return shortened


def _ensure_audio_cache_dir():
    try:
        os.makedirs(AUDIO_CACHE_PATH, exist_ok=True)
    except Exception:
        pass


def _audio_cache_key(text: str, voice: str, rate: int) -> str:
    key_src = f"{voice}|{rate}|{text}".encode("utf-8")
    return hashlib.sha256(key_src).hexdigest()


def _audio_cache_paths_for_key(key: str) -> tuple:
    p = Path(AUDIO_CACHE_PATH)
    return (p / f"{key}.mp3", p / f"{key}.json")


def _get_audio_cache_lock(key: str):
    # return an asyncio.Lock for the given key (create if necessary)
    if key not in _audio_cache_locks:
        _audio_cache_locks[key] = asyncio.Lock()
    return _audio_cache_locks[key]


def _read_cached_audio(key: str):
    """Return bytes or None if not found/expired."""
    global AUDIO_CACHE_HITS, AUDIO_CACHE_MISSES
    if not AUDIO_CACHE_ENABLED:
        return None
    _ensure_audio_cache_dir()
    mp3_path, meta_path = _audio_cache_paths_for_key(key)
    try:
        if not mp3_path.exists():
            AUDIO_CACHE_MISSES += 1
            return None
        # check TTL if metadata exists
        if meta_path.exists():
            try:
                import json
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                ts = meta.get("ts", 0)
                if AUDIO_CACHE_TTL and (time.time() - ts) > AUDIO_CACHE_TTL:
                    # expired
                    try:
                        mp3_path.unlink()
                    except Exception:
                        pass
                    try:
                        meta_path.unlink()
                    except Exception:
                        pass
                    AUDIO_CACHE_MISSES += 1
                    return None
            except Exception:
                pass
        # read bytes
        with open(mp3_path, "rb") as fh:
            data = fh.read()
        AUDIO_CACHE_HITS += 1
        return data
    except Exception:
        AUDIO_CACHE_MISSES += 1
        return None


def _write_audio_cache(key: str, data: bytes):
    _ensure_audio_cache_dir()
    mp3_path, meta_path = _audio_cache_paths_for_key(key)
    try:
        # atomic write
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        with open(tmp.name, "wb") as fh:
            fh.write(data)
        os.replace(tmp.name, mp3_path)
        # write metadata
        meta = {"ts": time.time()}
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        return True
    except Exception:
        try:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
        except Exception:
            pass
        return False


async def tts_save_ssml(text: str, voice: str, rate_percent: int, out_path: str | None = None, return_bytes: bool = False):
    import xml.sax.saxutils as sax
    escaped = sax.escape(text)
    ssml = (
        f"<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\" xml:lang=\"fr-FR\">"
        f"<voice name=\"{voice}\"><prosody rate=\"{rate_percent}%\">{escaped}</prosody></voice></speak>"
    )
    communicate = edge_tts.Communicate(ssml, voice)

    if return_bytes:
        import tempfile
        tmpf = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tmpf.close()
        try:
            await communicate.save(tmpf.name)
            with open(tmpf.name, 'rb') as fh:
                data = fh.read()
            return data
        finally:
            try:
                os.remove(tmpf.name)
            except Exception:
                pass

    if not return_bytes:
        if not out_path:
            raise ValueError("out_path must be specified when return_bytes is False")
        await communicate.save(out_path)
        return None

async def generate_tts_synced(text: str, voice: str, target_duration: float, out_path: str | None, sem: asyncio.BoundedSemaphore, timeout_per_attempt: int = 30, max_attempts: int = 4, return_bytes: bool = False):
    """Async version: generate TTS trying to match the target_duration by adjusting rate and shortening.

    If return_bytes is True, the function returns audio bytes in the returned dict under key 'audio_bytes' and does not write a block file to disk.
    """
    rate = compute_rate_percent(text, target_duration)
    shortened = False
    current_text = text

    # Audio cache key (based on voice, rate and text). If enabled and return_bytes requested, try serving from cache
    key = _audio_cache_key(current_text, voice, rate)
    if return_bytes and AUDIO_CACHE_ENABLED:
        # quick check without lock
        cached = _read_cached_audio(key)
        if cached:
            try:
                a_seg = AudioSegment.from_file(io.BytesIO(cached), format="mp3")
                dur = len(a_seg) / 1000.0
            except Exception:
                dur = None
            return {"duration": dur, "rate": rate, "attempts": 0, "shortened": False, "audio_bytes": cached}

    # If caller expects a file on disk, try to serve from cache to avoid TTS call
    if (not return_bytes) and AUDIO_CACHE_ENABLED and out_path:
        cached = _read_cached_audio(key)
        if cached:
            try:
                with open(out_path, 'wb') as fh:
                    fh.write(cached)
                try:
                    a_seg = AudioSegment.from_file(io.BytesIO(cached), format="mp3")
                    dur = len(a_seg) / 1000.0
                except Exception:
                    dur = None
                return {"duration": dur, "rate": rate, "attempts": 0, "shortened": False}
            except Exception:
                pass

    # If not in cache and return_bytes requested, acquire per-key lock when generating the audio to avoid duplicate work
    lock = None
    if return_bytes and AUDIO_CACHE_ENABLED:
        lock = _get_audio_cache_lock(key)


    last_dur = None
    for attempt in range(1, max_attempts + 1):
        try:
            # If we have an audio cache lock, coordinate cache-aware generation
            if lock is not None:
                async with lock:
                    # double-check cache (someone else may have written it while waiting for lock)
                    cached = _read_cached_audio(key)
                    if cached:
                        audio_bytes = cached
                    else:
                        async with sem:
                            audio_bytes = await asyncio.wait_for(tts_save_ssml(current_text, voice, rate, None, return_bytes=True), timeout=timeout_per_attempt)
                            # store in cache (background write ok)
                            try:
                                await asyncio.to_thread(_write_audio_cache, key, audio_bytes)
                            except Exception:
                                pass
            else:
                async with sem:
                    if return_bytes:
                        audio_bytes = await asyncio.wait_for(tts_save_ssml(current_text, voice, rate, None, return_bytes=True), timeout=timeout_per_attempt)
                    else:
                        await asyncio.wait_for(tts_save_ssml(current_text, voice, rate, out_path), timeout=timeout_per_attempt)
            # measure
            try:
                if return_bytes and audio_bytes:
                    a_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
                    dur = len(a_seg) / 1000.0
                else:
                    from moviepy import AudioFileClip
                    a = AudioFileClip(out_path)
                    dur = a.duration
                    a.close()
            except Exception:
                dur = None

            # If we generated a file on disk, try to cache it for future calls
            if (not return_bytes) and AUDIO_CACHE_ENABLED and out_path and os.path.exists(out_path):
                try:
                    with open(out_path, 'rb') as fh:
                        d = fh.read()
                    # write cache in background thread
                    await asyncio.to_thread(_write_audio_cache, key, d)
                except Exception:
                    pass
        except asyncio.TimeoutError:
            dur = None
        except Exception:
            dur = None

        last_dur = dur
        if dur is not None:
            diff_ratio = abs(dur - target_duration) / max(target_duration, 0.01)
            if diff_ratio <= 0.18:
                res = {"duration": dur, "rate": rate, "attempts": attempt, "shortened": shortened}
                if return_bytes and audio_bytes:
                    res["audio_bytes"] = audio_bytes
                return res
            if dur > 0.1 and target_duration > 0.1:
                new_rate = int(rate * (dur / target_duration))
                new_rate = max(MIN_RATE, min(MAX_RATE, new_rate))
                if new_rate != rate:
                    rate = new_rate
                    continue
        rate = min(MAX_RATE, int(rate * 1.3))

    # try shorten if still bad
    try:
        if return_bytes and audio_bytes:
            a_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            last_dur = len(a_seg) / 1000.0
        else:
            from moviepy import AudioFileClip
            a = AudioFileClip(out_path)
            last_dur = a.duration
            a.close()
    except Exception:
        last_dur = None

    if last_dur and last_dur > target_duration and rate >= MAX_RATE:
        shortened_text = shorten_text_to_fit(text, target_duration)
        if shortened_text != text:
            shortened = True
            current_text = shortened_text
            rate = compute_rate_percent(current_text, target_duration)
            for attempt in range(1, max_attempts + 1):
                try:
                    async with sem:
                        if return_bytes:
                            audio_bytes = await asyncio.wait_for(tts_save_ssml(current_text, voice, rate, None, return_bytes=True), timeout=timeout_per_attempt)
                        else:
                            await asyncio.wait_for(tts_save_ssml(current_text, voice, rate, out_path), timeout=timeout_per_attempt)
                    if return_bytes and audio_bytes:
                        a_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
                        dur = len(a_seg) / 1000.0
                    else:
                        from moviepy import AudioFileClip
                        a = AudioFileClip(out_path)
                        dur = a.duration
                        a.close()
                except Exception:
                    dur = None
                if dur is not None:
                    diff_ratio = abs(dur - target_duration) / max(target_duration, 0.01)
                    if diff_ratio <= 0.18:
                        res = {"duration": dur, "rate": rate, "attempts": attempt + max_attempts, "shortened": shortened}
                        if return_bytes and audio_bytes:
                            res["audio_bytes"] = audio_bytes
                        return res
                    new_rate = int(rate * (dur / target_duration))
                    new_rate = max(MIN_RATE, min(MAX_RATE, new_rate))
                    if new_rate == rate:
                        break
                    rate = new_rate
    # If we have audio bytes but have not computed a duration yet, try to compute it
    if return_bytes and 'audio_bytes' in locals() and (not ('last_dur' in locals()) or last_dur is None):
        try:
            a_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            last_dur = len(a_seg) / 1000.0
        except Exception:
            last_dur = None

    res = {"duration": last_dur if 'last_dur' in locals() else None, "rate": rate, "attempts": max_attempts, "shortened": shortened}
    if return_bytes and 'audio_bytes' in locals():
        res["audio_bytes"] = audio_bytes
    return res


# Existing functions from dub2

def extract_audio(video_path, audio_path):
    """Extrait l'audio de la vidéo."""
    print(f"1. Extraction de l'audio depuis {video_path}...")
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None) # logger=None pour réduire les logs inutiles
        clip.close()
        print("   -> Audio extrait avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'extraction : {e}")
        raise


def transcribe_with_whisper_cpp(audio_path):
    """Transcrit l'audio localement en privilégiant whisper.cpp (fallback: faster_whisper, openai-whisper)."""
    import traceback
    print("2. Transcription locale (whisper.cpp preferred)...")

    # 1) whisper.cpp (binding Python) - API variable selon l'implémentation
    try:
        import whisper_cpp
        print("   -> Utilisation de whisper_cpp")
        # support a few possible APIs for different bindings
        if hasattr(whisper_cpp, "transcribe"):
            result = whisper_cpp.transcribe(audio_path)
            transcript = result if isinstance(result, str) else result.get("text", "")
        elif hasattr(whisper_cpp, "WhisperModel"):
            model = whisper_cpp.WhisperModel()
            transcript = model.transcribe(audio_path)
        elif hasattr(whisper_cpp, "Whisper"):
            model = whisper_cpp.Whisper()
            transcript = model.transcribe(audio_path)
        else:
            raise RuntimeError("API de whisper_cpp non reconnue")

        with open(TEXT_OUTPUT, "w", encoding="utf-8") as f:
            f.write(transcript)
        print("   -> Transcription (whisper.cpp) terminée.")
        return transcript
    except Exception as e_wc:
        print(f"   -> whisper_cpp indisponible ou erreur: {e_wc}")
        # 2) faster_whisper fallback
        try:
            from faster_whisper import WhisperModel
            print("   -> Utilisation de faster_whisper en fallback")
            model = WhisperModel("small", device="cpu")
            segments, _ = model.transcribe(audio_path)
            transcript = "\n".join([s.text for s in segments])

            with open(TEXT_OUTPUT, "w", encoding="utf-8") as f:
                f.write(transcript)
            print("   -> Transcription (faster_whisper) terminée.")
            return transcript
        except Exception as e_fw:
            print(f"   -> faster_whisper indisponible ou erreur: {e_fw}")
            # 3) openai-whisper final fallback
            try:
                import whisper
                print("   -> Utilisation de openai-whisper en dernier recours")
                model = whisper.load_model("small")
                result = model.transcribe(audio_path)
                transcript = result.get("text", "")

                with open(TEXT_OUTPUT, "w", encoding="utf-8") as f:
                    f.write(transcript)
                print("   -> Transcription (openai-whisper) terminée.")
                return transcript
            except Exception as e_w:
                print(f"   -> Tous les fallback ont échoué: {e_w}")
                traceback.print_exc()
                raise


def translate_with_llama(text):
    """Traduit le texte via Groq en essayant une liste de modèles disponibles.

    La fonction parcourt `GROQ_MODEL_PREFERENCES` et tente chaque modèle à tour de rôle. En cas de
    rate-limit (429) pour un modèle, elle passe au suivant. Si tous les modèles Groq échouent, on
    bascule vers GoogleTranslator local (deep_translator) pour garantir une réponse.
    """
    print("3. Traduction avec Groq (liste de modèles)...")
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
Tu es un expert en doublage. Traduis le texte suivant de l'anglais vers le français.
Adapte le ton et essaie de garder une longueur de phrase compatible avec le timing vidéo.

Texte :
{text}

Retourne UNIQUEMENT la traduction.
"""

    last_exc = None
    # Shortcut: if the text is too long for Groq, prefer Google to avoid token/quota errors
    if GROQ_MAX_CHARS is not None and len(text) > GROQ_MAX_CHARS:
        print(f"   -> Texte long ({len(text)} chars) > GROQ_MAX_CHARS ({GROQ_MAX_CHARS}). Utilisation de Google pour éviter les limites de tokens.")
        return translate_with_google(text)

    for model in GROQ_MODEL_PREFERENCES:
        try:
            print(f"   -> Tentative avec le modèle: {model}...")
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.3,
            )
            french_text = chat_completion.choices[0].message.content
            with open(FRENCH_TEXT, "w", encoding="utf-8") as f:
                f.write(french_text)
            print(f"   -> Traduction terminée (modèle: {model}).")
            return french_text
        except Exception as e:
            last_exc = e
            _msg = str(e).lower()
            # détection de rate-limit via message ou HTTPError
            if '429' in _msg or 'rate limit' in _msg or 'rate_limit' in _msg or 'rate_limit_exceeded' in _msg:
                print(f"   -> Rate limit détecté pour {model}, basculement sur le modèle suivant...")
                continue
            # détection d'erreurs liées aux tokens / quotas / longueur de contexte
            if 'token' in _msg or 'tokens' in _msg or 'token limit' in _msg or 'max tokens' in _msg or 'quota' in _msg or 'context length' in _msg or 'context' in _msg:
                print(f"   -> Limite de tokens détectée pour {model} (message: {e}). Basculement immédiat vers Google Translator.")
                return translate_with_google(text)
            try:
                from httpx import HTTPStatusError
                if isinstance(e, HTTPStatusError) and e.response is not None and e.response.status_code == 429:
                    print(f"   -> HTTP 429 reçu pour {model}, basculer sur le suivant...")
                    continue
            except Exception:
                pass
            # autres erreurs: on logue et on essaie le modèle suivant
            print(f"   -> Erreur avec {model}: {e} (essai du modèle suivant)")
            continue

    # si on arrive ici, tous les modèles Groq ont échoué ou été rate-limited
    print("   -> Tous les modèles Groq ont échoué ou sont limités. Basculement vers Google Translator.")
    return translate_with_google(text)


def translate_with_google(text):
    """Fallback local via deep_translator GoogleTranslator."""
    print("3b. Traduction via GoogleTranslator (deep_translator)...")
    try:
        from deep_translator import GoogleTranslator as DT
        res = DT(source='auto', target='fr').translate(text)
        with open(FRENCH_TEXT, "w", encoding="utf-8") as f:
            f.write(res)
        print("   -> Traduction (Google) terminée.")
        return res
    except Exception as e:
        print(f"   -> Erreur GoogleTranslator: {e}")
        raise


# --- Translation cache helpers ---
def _load_translation_cache():
    global _translation_cache
    with _translation_cache_lock:
        if _translation_cache is not None:
            return _translation_cache
        try:
            if os.path.exists(TRANSLATION_CACHE_PATH):
                with open(TRANSLATION_CACHE_PATH, "r", encoding="utf-8") as fh:
                    _translation_cache = json.load(fh)
            else:
                _translation_cache = {}
        except Exception:
            _translation_cache = {}
        return _translation_cache


def _save_translation_cache():
    with _translation_cache_lock:
        try:
            with open(TRANSLATION_CACHE_PATH, "w", encoding="utf-8") as fh:
                json.dump(_translation_cache, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Erreur en sauvegardant le cache de traduction: {e}")


def _make_cache_key(text: str, mode: str):
    # normalize whitespace and use mode to separate caches
    k = "|".join([mode or "auto", " ".join(text.split())])
    return k


def get_cached_translation(text: str, mode: str = 'auto'):
    global TRANSLATION_CACHE_HITS, TRANSLATION_CACHE_MISSES
    if not TRANSLATION_CACHE_ENABLED:
        return None
    cache = _load_translation_cache()
    key = _make_cache_key(text, mode)
    v = cache.get(key)
    if not v:
        with _translation_cache_lock:
            TRANSLATION_CACHE_MISSES += 1
        return None
    ts = v.get("ts", 0)
    if TRANSLATION_CACHE_TTL is not None and (time.time() - ts) > TRANSLATION_CACHE_TTL:
        # expired
        with _translation_cache_lock:
            TRANSLATION_CACHE_MISSES += 1
        return None
    with _translation_cache_lock:
        TRANSLATION_CACHE_HITS += 1
    return v.get("translation")


def set_cached_translation(text: str, translation: str, mode: str = 'auto'):
    if not TRANSLATION_CACHE_ENABLED:
        return
    cache = _load_translation_cache()
    key = _make_cache_key(text, mode)
    cache[key] = {"translation": translation, "ts": int(time.time())}
    _save_translation_cache()



def translate_text(text, mode='auto'):
    """Choix de traducteur: 'auto', 'groq', 'google'.

    - 'auto': tente Groq, bascule vers Google en cas d'erreur.
    - 'groq': force Groq.
    - 'google': force GoogleTranslator local.
    Utilise un cache persistant pour réduire les appels externes.
    """
    # Check cache first
    cached = get_cached_translation(text, mode)
    if cached:
        print("   -> Traduction issue du cache.")
        return cached

    # perform translation
    if mode == 'google':
        res = translate_with_google(text)
        set_cached_translation(text, res, mode)
        return res
    if mode == 'groq':
        res = translate_with_llama(text)
        set_cached_translation(text, res, mode)
        return res
    # auto: try groq then google
    try:
        res = translate_with_llama(text)
        set_cached_translation(text, res, 'auto')
        return res
    except Exception:
        res = translate_with_google(text)
        set_cached_translation(text, res, 'auto')
        return res


async def translate_text_async(text, mode='auto'):
    """Async wrapper qui exécute la traduction en thread pour ne pas bloquer l'event loop."""
    return await asyncio.to_thread(translate_text, text, mode)


async def generate_french_audio(text, output_path):
    """Génère l'audio Français via Edge-TTS."""
    print("4. Génération de l'audio Français (Edge-TTS)...")
    # Utilisation d'une voix homme ou femme standard
    voice = "fr-FR-DeniseNeural"

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    print(f"   -> Audio généré : {output_path}")


def merge_audio_video(video_path, new_audio_path, output_path):
    """Assemble la vidéo originale avec le nouvel audio."""
    print("5. Assemblage final de la vidéo...")
    video = None
    new_audio = None
    final_video = None
    try:
        video = VideoFileClip(video_path)
        new_audio = AudioFileClip(new_audio_path)

        # Gestion simple de la durée : on ajuste l'audio à la durée de la vidéo
        final_audio = new_audio
        try:
            if new_audio.duration > video.duration:
                # set_duration tranche ou étend l'audio de façon sûre
                final_audio = new_audio.set_duration(video.duration)
        except Exception:
            # En cas d'erreur d'accès à duration, on continue sans couper
            pass

        # moviepy v2 uses `with_audio` to attach an audio clip
        final_video = video.with_audio(final_audio)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)

        print("   -> Vide9o finale sauvegarde9e.")
    except Exception as e:
        print(f"Erreur lors de l'assemblage : {e}")
        raise
    finally:
        # S'assurer que tous les clips sont ferme9s pour libe9rer les fichiers
        for clip in (final_video, video, new_audio):
            try:
                if clip is not None:
                    clip.close()
            except Exception:
                pass


def clean_up():
    """Nettoie les fichiers temporaires."""
    files = [TEMP_AUDIO_EN, TEXT_OUTPUT, FRENCH_TEXT, TEMP_DIR]
    for f in files:
        try:
            if os.path.exists(f):
                if os.path.isdir(f):
                    import shutil
                    try:
                        shutil.rmtree(f)
                    except PermissionError:
                        print(f"Warning: unable to remove directory {f} (file in use). Skipping")
                    except Exception as e:
                        print(f"Warning: error removing directory {f}: {e}")
                else:
                    try:
                        os.remove(f)
                    except PermissionError:
                        print(f"Warning: unable to remove file {f} (file in use). Skipping")
                    except Exception as e:
                        print(f"Warning: error removing file {f}: {e}")
        except Exception as e:
            print(f"Warning during cleanup check for {f}: {e}")


# --- EXÉCUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doublage vidéo synchronisé par segments SRT")
    parser.add_argument("--translator", choices=["auto", "groq", "google"], default="auto", help="Choisir le moteur de traduction (auto: groq puis google fallback)")
    parser.add_argument("--concurrency", "-c", type=int, default=6, help="Nombre max de tâches TTS concurrentes")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout (s) par tentative TTS")
    parser.add_argument("--max-attempts", type=int, default=4, help="Nombre max d'essais pour ajuster le rate / raccourcir le texte")
    parser.add_argument("--test-translate", action="store_true", help="Tester uniquement la traduction (utile pour vérifier les basculements de modèle)")
    parser.add_argument("--groq-max-chars", type=int, default=None, help="Forcer l'utilisation de Google si le texte a plus de N caractères (évite erreurs de token)")
    parser.add_argument("--batch-duration", type=float, default=12.0, help="Durée maximale (secondes) d'un bloc TTS (regroupement de segments)")
    parser.add_argument("--no-batch", action="store_true", help="Désactiver le batching et générer chaque segment indépendamment")
    parser.add_argument("--block-concurrency", type=int, default=4, help="Nombre de blocs TTS traités en parallèle")
    parser.add_argument("--block-timeout", type=int, default=120, help="Timeout (s) pour le traitement d'un bloc TTS (génération + split)")
    parser.add_argument("--no-cache", action="store_true", help="Désactiver le cache de traduction")
    parser.add_argument("--cache-ttl", type=int, default=None, help="Durée (en secondes) du TTL pour le cache (défaut: valeur intégrée ou env)")
    parser.add_argument("--cache-path", type=str, default=None, help="Chemin du fichier cache (défaut: translation_cache.json)")
    parser.add_argument("--no-audio-cache", action="store_true", help="Désactiver le cache audio (texte->mp3)")
    parser.add_argument("--audio-cache-path", type=str, default=None, help="Chemin pour stocker le cache audio (défaut: ./audio_cache)")
    parser.add_argument("--audio-cache-ttl", type=int, default=None, help="TTL (s) pour le cache audio (défaut: valeur intégrée ou env)")
    args = parser.parse_args()
    # configure cache settings from args
    if args.no_cache:
        TRANSLATION_CACHE_ENABLED = False
    if args.cache_ttl is not None:
        TRANSLATION_CACHE_TTL = int(args.cache_ttl)
    if args.cache_path:
        TRANSLATION_CACHE_PATH = args.cache_path
    if args.groq_max_chars is not None:
        GROQ_MAX_CHARS = int(args.groq_max_chars)
    if args.no_audio_cache:
        AUDIO_CACHE_ENABLED = False
    if args.audio_cache_path:
        AUDIO_CACHE_PATH = args.audio_cache_path
    if args.audio_cache_ttl is not None:
        AUDIO_CACHE_TTL = int(args.audio_cache_ttl)
    # reset in-memory cache so changes take effect
    with _translation_cache_lock:
        _translation_cache = None

    # Vérification que le fichier vidéo existe
    if args.test_translate:
        sample = "Hello, this is a short test sentence to verify translation model switching."
        print("-- Test translation (mode: %s) --" % args.translator)
        try:
            res = translate_text(sample, args.translator)
            print("Résultat :\n", res)
        except Exception as e:
            print("Erreur lors du test de traduction :", e)
        raise SystemExit(0)
    if not os.path.exists(INPUT_VIDEO):
        print(f"Erreur : Le fichier '{INPUT_VIDEO}' est introuvable dans le répertoire actuel.")
    else:
        try:
            # Si un fichier SRT est présent, on utilise la méthode synchronisée par segments
            srt_path = "subtitle.srt"
            if os.path.exists(srt_path):
                print("SRT detecte - generation segmentee et synchronisee...")
                subs = parse_srt(srt_path)
                os.makedirs(TEMP_DIR, exist_ok=True)
                voice = "fr-FR-HenriNeural"
                segments_audio = []

                concurrency = args.concurrency
                sem = asyncio.BoundedSemaphore(concurrency)

                async def produce_all():
                    # 0) Traduire tous les segments en parallèle (thread pool pour ne pas bloquer)
                    print("0. Traduction des segments (concurrent)...")
                    translate_tasks = [asyncio.create_task(translate_text_async(s["text"], args.translator)) for s in subs]
                    translations = await asyncio.gather(*translate_tasks)

                    # 1) Regrouper en blocs si batching activé
                    blocks = []
                    if args.no_batch:
                        # chaque segment est un bloc
                        for i, s in enumerate(subs):
                            d = max(0.1, s["end"] - s["start"])
                            blocks.append({"indices": [i], "texts": [translations[i]], "duration": d})
                    else:
                        cur = {"indices": [], "texts": [], "duration": 0.0}
                        for i, s in enumerate(subs):
                            d = max(0.1, s["end"] - s["start"])
                            if cur["indices"] and (cur["duration"] + d) > args.batch_duration:
                                blocks.append(cur)
                                cur = {"indices": [], "texts": [], "duration": 0.0}
                            cur["indices"].append(i)
                            cur["texts"].append(translations[i])
                            cur["duration"] += d
                        if cur["indices"]:
                            blocks.append(cur)

                    # 2) Générer un fichier par bloc puis découper en segments
                    results = [None] * len(subs)
                    report = []

                    # Run blocks concurrently: schedule tasks and split in threads to avoid blocking the event loop
                    results = [None] * len(subs)
                    report = []

                    # helper to run blocking splitting in a thread
                    def _split_block_to_segments(block_path, block, b_idx):
                        local_report = []
                        local_results = {}
                        a_block = None
                        try:
                            a_block = AudioFileClip(block_path)
                            offset = 0.0
                            for j, idx in enumerate(block["indices"]):
                                seg_dur = max(0.1, subs[idx]["end"] - subs[idx]["start"])
                                start = offset
                                end = min(offset + seg_dur, a_block.duration)
                                seg_path = os.path.join(TEMP_DIR, f"seg_{idx:04d}.mp3")
                                sub = a_block.subclip(start, end)
                                sub.write_audiofile(seg_path, logger=None)
                                try:
                                    sub.close()
                                except Exception:
                                    pass
                                local_results[idx] = {"translated": translations[idx], "out_path": seg_path, "meta": {"from_block": b_idx, "block_duration": a_block.duration}}
                                local_report.append({"index": idx, "status": "ok", "block": b_idx, "out_path": seg_path})
                                offset += seg_dur
                        except Exception as e:
                            for idx in block["indices"]:
                                local_results[idx] = {"translated": translations[idx], "out_path": None, "meta": {"error": str(e)}}
                                local_report.append({"index": idx, "status": "failed_split", "error": str(e)})
                        finally:
                            try:
                                if a_block is not None:
                                    a_block.close()
                            except Exception:
                                pass
                        return local_results, local_report

                    def _split_block_bytes_to_segments(audio_bytes, block, b_idx):
                        """Split in-memory mp3 bytes into per-segment mp3 files using pydub."""
                        local_report = []
                        local_results = {}
                        try:
                            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
                            offset_ms = 0
                            for j, idx in enumerate(block["indices"]):
                                seg_dur_ms = int(max(0.1, subs[idx]["end"] - subs[idx]["start"]) * 1000)
                                start_ms = offset_ms
                                end_ms = min(offset_ms + seg_dur_ms, len(audio))
                                chunk = audio[start_ms:end_ms]
                                seg_path = os.path.join(TEMP_DIR, f"seg_{idx:04d}.mp3")
                                # export
                                chunk.export(seg_path, format="mp3")
                                local_results[idx] = {"translated": translations[idx], "out_path": seg_path, "meta": {"from_block": b_idx, "block_duration": len(audio) / 1000.0}}
                                local_report.append({"index": idx, "status": "ok", "block": b_idx, "out_path": seg_path})
                                offset_ms += seg_dur_ms
                        except Exception as e:
                            for idx in block["indices"]:
                                local_results[idx] = {"translated": translations[idx], "out_path": None, "meta": {"error": str(e)}}
                                local_report.append({"index": idx, "status": "failed_split", "error": str(e)})
                        return local_results, local_report

                    async def process_block(b_idx, block):
                        block_path = os.path.join(TEMP_DIR, f"block_{b_idx:04d}.mp3")
                        # Check skip/resume quickly
                        all_ok = True
                        for idx in block["indices"]:
                            seg_path = os.path.join(TEMP_DIR, f"seg_{idx:04d}.mp3")
                            if not os.path.exists(seg_path):
                                all_ok = False
                                break
                            try:
                                acheck = AudioFileClip(seg_path)
                                seg_dur = max(0.1, subs[idx]["end"] - subs[idx]["start"])
                                if abs(acheck.duration - seg_dur) / max(seg_dur, 0.01) > 0.18:
                                    all_ok = False
                                acheck.close()
                                if not all_ok:
                                    break
                            except Exception:
                                all_ok = False
                                break

                        if all_ok:
                            return {idx: {"translated": translations[idx], "out_path": os.path.join(TEMP_DIR, f"seg_{idx:04d}.mp3"), "meta": {"skipped": True}} for idx in block["indices"]}, [{"index": idx, "status": "skipped"} for idx in block["indices"]]

                        combined_text = " ".join(block["texts"]).strip()
                        target_duration = block["duration"]

                        try:
                            # request in-memory audio bytes to avoid writing a block file
                            meta = await asyncio.wait_for(generate_tts_synced(combined_text, voice, target_duration, None, sem, timeout_per_attempt=args.timeout, max_attempts=args.max_attempts, return_bytes=True), timeout=args.block_timeout)
                        except asyncio.TimeoutError:
                            return ({idx: {"translated": translations[idx], "out_path": None, "meta": {"error": "block_timeout"}} for idx in block["indices"]}, [{"index": idx, "status": "failed", "reason": "timeout"} for idx in block["indices"]])
                        except Exception as e:
                            return ({idx: {"translated": translations[idx], "out_path": None, "meta": {"error": str(e)}} for idx in block["indices"]}, [{"index": idx, "status": "failed", "reason": str(e)} for idx in block["indices"]])

                        if not meta.get("duration"):
                            return ({idx: {"translated": translations[idx], "out_path": None, "meta": {"error": "block_failed"}} for idx in block["indices"]}, [{"index": idx, "status": "failed", "reason": "block_failed"} for idx in block["indices"]])

                        # If audio bytes were returned, split in-memory
                        if meta.get("audio_bytes"):
                            local_results, local_report = await asyncio.to_thread(_split_block_bytes_to_segments, meta.get("audio_bytes"), block, b_idx)
                        else:
                            # No bytes returned. If audio cache is enabled, attempt to fetch a cached block for the combined_text
                            if AUDIO_CACHE_ENABLED:
                                try:
                                    k = _audio_cache_key(combined_text, voice, meta.get('rate', rate))
                                    cached = _read_cached_audio(k)
                                    if cached:
                                        local_results, local_report = await asyncio.to_thread(_split_block_bytes_to_segments, cached, block, b_idx)
                                    else:
                                        local_results, local_report = await asyncio.to_thread(_split_block_to_segments, block_path, block, b_idx)
                                except Exception:
                                    local_results, local_report = await asyncio.to_thread(_split_block_to_segments, block_path, block, b_idx)
                            else:
                                local_results, local_report = await asyncio.to_thread(_split_block_to_segments, block_path, block, b_idx)

                        try:
                            if not meta.get("audio_bytes") and os.path.exists(block_path):
                                os.remove(block_path)
                        except Exception:
                            pass

                        return local_results, local_report

                    # schedule all block tasks with limited concurrency
                    block_sem = asyncio.BoundedSemaphore(args.block_concurrency)

                    async def _worker(b_idx, block):
                        async with block_sem:
                            return await process_block(b_idx, block)

                    tasks = []
                    for b_idx, block in enumerate(blocks):
                        tasks.append(asyncio.create_task(_worker(b_idx, block)))

                    completed = 0
                    if _tqdm:
                        pbar = _tqdm(total=len(tasks), desc="Génération blocs TTS")
                    else:
                        pbar = None

                    for fut in asyncio.as_completed(tasks):
                        try:
                            local_results, local_report = await fut
                        except Exception as e:
                            # task failed
                            local_results = {}
                            local_report = []
                        for k, v in local_results.items():
                            results[k] = v
                        report.extend(local_report)
                        completed += 1
                        if pbar:
                            pbar.update(1)

                    if pbar:
                        pbar.close()

                    # Save report
                    try:
                        with open(os.path.join(TEMP_DIR, "segments_report.json"), "w", encoding="utf-8") as fh:
                            json.dump(report, fh, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                    return results

                results = asyncio.run(produce_all())

                for i, r in enumerate(results):
                    s = subs[i]
                    segments_audio.append({
                        "path": r["out_path"],
                        "start": s["start"],
                        "duration": max(0.1, s["end"] - s["start"]),
                        "translated": r["translated"],
                        "meta": r["meta"],
                    })

                # Compose audio segments
                from moviepy.audio.AudioClip import CompositeAudioClip
                audio_clips = []
                for seg in segments_audio:
                    a = AudioFileClip(seg["path"])
                    try:
                        if hasattr(a, "with_duration") and a.duration > seg["duration"]:
                            a = a.with_duration(seg["duration"])
                    except Exception:
                        pass
                    try:
                        if hasattr(a, "with_start"):
                            a = a.with_start(seg["start"])
                    except Exception:
                        pass
                    audio_clips.append(a)

                composite = CompositeAudioClip(audio_clips)
                video = VideoFileClip(INPUT_VIDEO)
                final_video = video.with_audio(composite)
                final_video.write_videofile(OUTPUT_VIDEO, codec="libx264", audio_codec="aac", logger=None)
                print("\n✅ SUCCÈS ! La vidéo doublée synchronisee est 'video_output_french_synced.mp4'")
            else:
                extract_audio(INPUT_VIDEO, TEMP_AUDIO_EN)
                original_text = transcribe_with_whisper_cpp(TEMP_AUDIO_EN)
                translated_text = translate_text(original_text, args.translator)
                asyncio.run(generate_french_audio(translated_text, "temp_audio_fr.mp3"))
                merge_audio_video(INPUT_VIDEO, "temp_audio_fr.mp3", OUTPUT_VIDEO)
                print("\n✅ SUCCÈS ! La vidéo doublée est 'video_output_french.mp4'")
        except Exception as e:
            print(f"\n❌ Le processus a échoué : {e}")
        finally:
            # Afficher le résumé du cache si activé
            try:
                if TRANSLATION_CACHE_ENABLED:
                    print('\n--- Translation cache summary ---')
                    print(f"Cache file: {TRANSLATION_CACHE_PATH}")
                    print(f"Hits: {TRANSLATION_CACHE_HITS}  Misses: {TRANSLATION_CACHE_MISSES}")
            except Exception:
                pass
            try:
                if AUDIO_CACHE_ENABLED:
                    print('\n--- Audio cache summary ---')
                    print(f"Cache dir: {AUDIO_CACHE_PATH}")
                    print(f"Hits: {AUDIO_CACHE_HITS}  Misses: {AUDIO_CACHE_MISSES}")
            except Exception:
                pass
            # Nettoyage
            clean_up()