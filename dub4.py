import os
import asyncio
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import edge_tts
import requests
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import argparse

# GUI imports
import threading
import sys
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
except Exception:
    tk = None

# 1. Chargement de la clé API depuis le fichier .env
load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("La clé API GROQ n'est pas définie dans le fichier .env")

# 2. Configuration des fichiers (modifiables via l'UI ou la CLI)
# Aucun chemin absolu par défaut : l'utilisateur choisit toujours les fichiers.
TEMP_AUDIO_FR = "temp_audio_fr.mp3"    # Audio français temporaire

# Voix et débit pour la synthèse vocale (lecture rapide et naturelle)
EDGE_TTS_VOICE = "fr-FR-HenriNeural"     # Voix masculine française
EDGE_TTS_RATE = "+10%"                   # Lecture très rapide

# Échantillon vocal extrait de la vidéo source (pour clonage éventuel)
VOICE_SAMPLE_DURATION_S = 12            # Secondes à extraire du début de la vidéo (6–30 s recommandé pour le clonage)
VOICE_SAMPLE_FILENAME = "voice_sample_source.wav"
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")  # Si défini, clonage vocal via ElevenLabs

def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60.0)
    h, m = divmod(m, 60.0)
    if h >= 1:
        return f"{int(h)} h {int(m)} min {s:0.2f} s"
    if m >= 1:
        return f"{int(m)} min {s:0.2f} s"
    return f"{s:0.2f} s"

def _print_timing_report(timings: dict):
    total = timings.get("total_s")
    print("\n⏱️  Chronométrage du doublage")
    if total is not None:
        print(f"   - Total : {_format_duration(total)}")
    for key, label in (
        ("translate_s", "Traduction"),
        ("tts_s", "Génération TTS"),
        ("merge_s", "Assemblage vidéo"),
        ("cleanup_s", "Nettoyage"),
    ):
        if key in timings:
            print(f"   - {label} : {_format_duration(timings[key])}")

def _parse_vtt_cues_with_time_strings(content: str):
    """Parse le contenu WEBVTT/SRT en liste de (start_str, end_str, text).
    Conserve les chaînes de temps telles quelles pour la réécriture.
    """
    lines = [ln.rstrip("\n") for ln in content.splitlines()]
    cues = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.upper().startswith("WEBVTT"):
            i += 1
            continue
        if line.isdigit() and i + 1 < n and "-->" in lines[i + 1]:
            i += 1
            line = lines[i].strip()
        if "-->" in line:
            left, right = line.split("-->", 1)
            start_str = left.strip()
            end_str = right.strip().split(" ")[0].strip()
            i += 1
            text_lines = []
            while i < n and lines[i].strip():
                if "-->" in lines[i]:
                    break
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines).strip()
            if text:
                cues.append((start_str, end_str, text))
        else:
            i += 1
    return cues


def _merge_adjacent_vtt_cues(cues: list) -> list:
    """Fusionne les cues consécutives lorsque end(cue i) == start(cue i+1).
    Réduit les micro-cues adjacentes en un seul bloc (même start que le premier, même end que le dernier).
    """
    if not cues:
        return []
    merged = []
    i = 0
    while i < len(cues):
        start_str, end_str, text = cues[i]
        j = i + 1
        while j < len(cues):
            try:
                end_ms = _parse_time_to_milliseconds(end_str)
                next_start_ms = _parse_time_to_milliseconds(cues[j][0])
            except ValueError:
                break
            if end_ms != next_start_ms:
                break
            _, end_str, next_text = cues[j]
            text = f"{text} {next_text}".strip()
            j += 1
        merged.append((start_str, end_str, text))
        i = j
    return merged


def _vtt_cues_to_content(cues: list, header: str = "WEBVTT") -> str:
    """Reconstruit le contenu WEBVTT à partir d'une liste de (start_str, end_str, text)."""
    if not cues:
        return header + "\n\n"
    blocks = [f"{start} --> {end}\n{text}" for start, end, text in cues]
    return header + "\n\n" + "\n\n".join(blocks) + "\n"


def _read_subtitle_text(path: str) -> str:
    """Lit le fichier de sous-titres, fusionne les cues adjacentes (end N = start N+1), retourne le VTT transformé."""
    print(f"1. Chargement des sous-titres : {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier de sous-titres '{path}' est introuvable.")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Détecter l'en-tête (WEBVTT ou vide pour SRT)
    header = "WEBVTT" if content.strip().upper().startswith("WEBVTT") else "WEBVTT"
    cues = _parse_vtt_cues_with_time_strings(content)
    if not cues:
        print("   -> Sous-titres chargés (aucun bloc).")
        return content
    original_count = len(cues)
    cues = _merge_adjacent_vtt_cues(cues)
    if len(cues) < original_count:
        print(f"   -> Sous-titres chargés et fusionnés : {original_count} → {len(cues)} blocs (cues adjacentes fusionnées).")
    else:
        print("   -> Sous-titres chargés.")
    return _vtt_cues_to_content(cues, header)

def _subtitle_path_from_video(video_path: str):
    """Retourne le chemin du fichier de sous-titres (même nom que la vidéo, extension .srt ou .vtt).
    Cherche d'abord .srt puis .vtt dans le même répertoire. Retourne None si aucun fichier trouvé.
    """
    if not video_path or not video_path.strip():
        return None
    base, _ = os.path.splitext(video_path)
    dirname = os.path.dirname(base)
    name = os.path.basename(base)
    base_path = os.path.join(dirname, name) if dirname else name
    for ext in (".srt", ".vtt"):
        path = base_path + ext
        if os.path.exists(path):
            return path
    return None

def _parse_time_to_seconds(time_str: str) -> float:
    """Convertit un timestamp SRT/VTT en secondes.

    Gère les formats :
    - hh:mm:ss,mmm
    - hh:mm:ss.mmm
    - mm:ss,mmm
    - mm:ss.mmm
    """
    return _parse_time_to_milliseconds(time_str) / 1000.0

def _parse_time_to_milliseconds(time_str: str) -> int:
    """Convertit un timestamp SRT/VTT en millisecondes (int) sans erreurs de float.

    Gère les formats :
    - hh:mm:ss,mmm
    - hh:mm:ss.mmm
    - mm:ss,mmm
    - mm:ss.mmm
    """
    time_str = time_str.strip()
    if not time_str:
        raise ValueError("Timestamp vide")

    # Uniformiser la séparation décimale
    time_str = time_str.replace(",", ".")
    hms, dot, ms = time_str.partition(".")
    parts = hms.split(":")

    # Accepter mm:ss ou hh:mm:ss
    if len(parts) == 2:
        h = 0
        m, s = parts
    elif len(parts) == 3:
        h, m, s = parts
    else:
        raise ValueError(f"Format de temps invalide: {time_str}")

    base_ms = (int(h) * 3600 + int(m) * 60 + int(s)) * 1000

    # Convertir la fraction en ms de manière robuste (1 chiffre = 100ms, 2 = 10ms, 3 = 1ms).
    frac_digits = "".join(ch for ch in ms if ch.isdigit()) if dot else ""
    if not frac_digits:
        frac_ms = 0
    elif len(frac_digits) <= 3:
        frac_ms = int(frac_digits.ljust(3, "0"))
    else:
        # Arrondi à la milliseconde la plus proche (au 4e chiffre).
        frac_ms = int(frac_digits[:3])
        if int(frac_digits[3]) >= 5:
            frac_ms += 1
        if frac_ms >= 1000:
            base_ms += 1000
            frac_ms = 0

    return base_ms + frac_ms

# Regex pour détecter la fin d'une phrase (.) ou (?) en fin de texte (espaces optionnels)
SENTENCE_END_RE = re.compile(r"[.?]\s*$")

def _ends_sentence(text: str) -> bool:
    """Vrai si le texte (après strip) se termine par . ou ?."""
    return bool(SENTENCE_END_RE.search(text.strip())) if text else False

def _merge_cues_until_sentence_end(cues: list) -> list:
    """Fusionne les blocs successifs en un seul tant que le texte ne se termine pas par . ou ?.

    Chaque bloc fusionné garde le start du premier et le end du dernier ; les textes sont
    concaténés sur une même ligne (séparés par un espace). Les timestamps ne sont pas modifiés.
    """
    if not cues:
        return []
    merged = []
    i = 0
    while i < len(cues):
        start_ms, end_ms, text = cues[i]
        j = i + 1
        while j < len(cues) and not _ends_sentence(text):
            _, next_end, next_text = cues[j]
            end_ms = next_end
            text = f"{text} {next_text}".strip()
            j += 1
        merged.append((start_ms, end_ms, text))
        i = j
    return merged

def _parse_subtitles_with_times(path: str):
    """Parse un fichier SRT/VTT et renvoie une liste de (start_ms, end_ms, text).

    Les blocs dont le texte ne se termine pas par . ou ? sont fusionnés avec les blocs
    suivants jusqu'à trouver une fin de phrase (regex SENTENCE_END_RE).
    """
    print(f"3. Analyse des sous-titres (avec temps) : {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier de sous-titres '{path}' est introuvable.")

    cues = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Ignorer l'en-tête WEBVTT
        if line.upper().startswith("WEBVTT"):
            i += 1
            continue

        # SRT: possible numéro de bloc
        if line.isdigit() and i + 1 < n and "-->" in lines[i + 1]:
            i += 1
            line = lines[i].strip()

        if "-->" in line:
            try:
                left, right = line.split("-->")
                start_ms = _parse_time_to_milliseconds(left.strip())
                end_ms = _parse_time_to_milliseconds(right.strip().split(" ")[0])
            except Exception:
                i += 1
                continue

            i += 1
            text_lines = []
            while i < n and lines[i].strip():
                # ignorer de futures lignes temps (sécurité)
                if "-->" in lines[i]:
                    break
                text_lines.append(lines[i].strip())
                i += 1

            text = " ".join(text_lines).strip()
            if text:
                cues.append((start_ms, end_ms, text))
        else:
            i += 1

    cues = _merge_cues_until_sentence_end(cues)
    print(f"   -> {len(cues)} blocs de sous-titres détectés (après fusion par fin de phrase).")
    return cues

def translate_with_llama(text: str, output_subtitle_path: str) -> str:
    print("2. Traduction des sous-titres avec Groq (Llama 4 Scout)...")
    client = Groq(api_key=GROQ_API_KEY)
    
    # prompt = f"""
    # Tu es un expert en sous-titres et en doublage.
    # Le texte ci-dessous est un fichier de sous-titres en anglais (par exemple SRT ou VTT).
    # Traduis UNIQUEMENT le texte visible à l'écran en français, en respectant STRICTEMENT les règles suivantes :
    # - tu dois conserver EXACTEMENT toutes les lignes contenant "-->" (les timestamps) SANS les modifier,
    # - tu ne changes PAS les numéros de blocs s'il y en a,
    # - tu ne modifies JAMAIS le format des timestamps ni leur position,
    # - SRT doit rester SRT, VTT doit rester VTT,
    # - tu remplaces simplement le texte dialogué par sa traduction française sur les lignes prévues pour le texte.
    
    # Fichier de sous-titres :
    # {text}
    
    # Retourne UNIQUEMENT le fichier de sous-titres complet en français, au même format, avec les mêmes timestamps.
    # """
    prompt = f"""
        You are an expert subtitle processor and translator.

        You are given an SRT or VTT subtitle file in English.
        Each subtitle block contains:
        - an index
        - a start and end timestamp
        - one sentence fragment or sentence

        Your task MUST follow these steps STRICTLY and IN ORDER:

        ────────────────────────────────
        STEP 1 — SENTENCE RECONSTRUCTION
        ────────────────────────────────

        Detect sentences that are split across multiple consecutive subtitle blocks.

        A sentence is considered "split" if:
        - it clearly continues grammatically or semantically in the next block
        - the sentence does NOT end with a strong punctuation mark (".", "?", "!")

        When a sentence is split across multiple blocks:
        - Merge them into a SINGLE subtitle entry
        - Concatenate the text in correct order with proper spacing
        - The new START timestamp = start time of the first block
        - The new END timestamp = end time of the last merged block
        - Remove the intermediate blocks completely

        Repeat this process until all sentences are complete.

        ────────────────────────────────
        STEP 2 — TRANSLATION
        ────────────────────────────────

        After ALL sentence reconstruction is finished:

        Translate EACH subtitle line into French.

        IMPORTANT RULES FOR TRANSLATION:
        - DO NOT modify timestamps
        - DO NOT merge or split any subtitles anymore
        - Translate line by line
        - Preserve the subtitle format (index, timestamp, text)
        - Use natural, professional French suitable for video subtitles
        - Keep the meaning accurate, not literal word-for-word
    Subtitle file : {text}
        ────────────────────────────────
        OUTPUT FORMAT
        ────────────────────────────────

        Return ONLY the final subtitle file in valid SRT format.

        Do NOT include:
        - explanations
        - comments
        - markdown
        - analysis
        - apologies

        Output must be directly usable as a subtitle file.
        return ONLY the complete subtitle file in french, with the same format.
        

    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/llama-4-scout-17b-16e-instruct", # Modèle spécifique de votre capture
            temperature=0.3,
        )

        french_subtitles = chat_completion.choices[0].message.content
        # Nettoyage : certains modèles ajoutent un préfixe explicatif, on le supprime si présent
        prefix = "Voici le fichier de sous-titres traduit en français :"
        if french_subtitles.strip().startswith(prefix):
            french_subtitles = french_subtitles.strip()[len(prefix):].lstrip("\n\r ")
        
        with open(output_subtitle_path, "w", encoding="utf-8") as f:
            f.write(french_subtitles)
        print(f"   -> Sous-titres traduits sauvegardés dans : {output_subtitle_path}")
        return french_subtitles
    except Exception as e:
        # Fallback si le modèle exact n'est pas encore dispo (au cas où)
        print(f"Attention : Erreur avec llama-4-scout ({e}). Tentative avec un modèle standard...")
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", # Alternative robuste
            temperature=0.3,
        )
        french_subtitles = chat_completion.choices[0].message.content
        prefix = "Voici le fichier de sous-titres traduit en français :"
        if french_subtitles.strip().startswith(prefix):
            french_subtitles = french_subtitles.strip()[len(prefix):].lstrip("\n\r ")
        with open(output_subtitle_path, "w", encoding="utf-8") as f:
            f.write(french_subtitles)
        print(f"   -> Sous-titres traduits (fallback) sauvegardés dans : {output_subtitle_path}")
        return french_subtitles

def _chunk_text_for_tts(text: str, max_chars: int = 1000):
    """Découpe le texte en morceaux compatibles avec les limites d'Edge-TTS.

    - Découpe d'abord par lignes (issues des sous-titres),
    """
    chunks = []
    current = []
    current_len = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        extra = len(line) + 1  # +1 pour un saut de ligne / espace
        if current_len + extra > max_chars and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += extra

    if current:
        chunks.append("\n".join(current))

    print(f"   -> Texte découpé en {len(chunks)} morceau(x) pour Edge-TTS.")
    return chunks


# def translate_first_cue(subtitle_path: str, output_subtitle_path: str) -> str:
#     """Traduit uniquement le texte du premier bloc de sous-titres en français.

#     - Conserve les timestamps et la structure du fichier.
#     - Écrit un nouveau fichier où seul le premier bloc textuel est remplacé par la traduction.
#     Retourne le contenu du fichier modifié.
#     """
#     print("Traduction : uniquement le premier bloc de sous-titres...")
#     if not os.path.exists(subtitle_path):
#         raise FileNotFoundError(f"Le fichier de sous-titres '{subtitle_path}' est introuvable.")

#     cues = _parse_subtitles_with_times(subtitle_path)
#     if not cues:
#         raise ValueError("Aucun bloc de sous-titres trouvé.")

#     first_text = cues[0][2]

#     client = Groq(api_key=GROQ_API_KEY)
#     prompt = f"""
#     Tu es un traducteur professionnel.
#     Traduits uniquement le texte suivant en français, sans ajouter d'autres explications ni modifier la ponctuation:

#     ---
#     {first_text}
#     ---

#     Retourne uniquement la traduction.
#     """

#     try:
#         chat_completion = client.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model="meta-llama/llama-4-scout-17b-16e-instruct",
#             temperature=0.0,
#         )
#         translated = chat_completion.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"Erreur Groq lors de la traduction du premier bloc: {e}. Tentative de fallback...")
#         chat_completion = client.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model="llama-3.3-70b-versatile",
#             temperature=0.0,
#         )
#         translated = chat_completion.choices[0].message.content.strip()

#     # Lire le fichier original et remplacer le premier bloc textuel
#     with open(subtitle_path, "r", encoding="utf-8") as f:
#         lines = [ln.rstrip("\n") for ln in f]

#     out_lines = []
#     i = 0
#     n = len(lines)
#     replaced = False
#     while i < n:
#         line = lines[i]
#         out_lines.append(line)
#         if not replaced and "-->" in line:
#             # on est sur la ligne de timestamps du premier bloc possible
#             # collect following text lines until blank
#             i += 1
#             # skip any blank lines immediately after timestamps
#             text_block_idx = i
#             text_block = []
#             while i < n and lines[i].strip():
#                 text_block.append(lines[i])
#                 i += 1

#             # write translated text (single or multiple lines)
#             for tline in translated.splitlines():
#                 out_lines.append(tline)

#             # if there was a blank line after original text, keep one
#             if i < n and not lines[i].strip():
#                 out_lines.append("")

#             replaced = True
#             continue
#         i += 1

#     result = "\n".join(out_lines)
#     with open(output_subtitle_path, "w", encoding="utf-8") as f:
#         f.write(result)

#     print(f"   -> Fichier avec premier bloc traduit sauvegardé dans : {output_subtitle_path}")
#     return result

async def generate_french_audio(text, output_path):
    """Génère l'audio Français via Edge-TTS (avec découpe en chunks pour ne rien couper)."""
    print("4. Génération de l'audio Français (Edge-TTS)...")
    voice = EDGE_TTS_VOICE
    rate = EDGE_TTS_RATE

    # On découpe le texte pour éviter les limitations de longueur côté service
    chunks = _chunk_text_for_tts(text, max_chars=10000)
    if not chunks:
        raise ValueError("Aucun texte fourni pour la synthèse vocale.")

    temp_files = []
    try:
        # 1) Générer un mp3 par chunk
        for idx, chunk in enumerate(chunks):
            temp_path = f"{output_path}.part{idx}.mp3"
            temp_files.append(temp_path)
            communicate = edge_tts.Communicate(chunk, voice, rate=rate)
            await communicate.save(temp_path)

        # 2) Concaténer tous les morceaux avec pydub
        combined = None
        for idx, temp_path in enumerate(temp_files):
            seg = AudioSegment.from_file(temp_path, format="mp3")
            if combined is None:
                combined = seg
            else:
                combined += seg

        if combined is None:
            raise RuntimeError("Échec de la concaténation audio : aucun segment généré.")

        combined.export(output_path, format="mp3")
        print(f"   -> Audio généré : {output_path}")
    finally:
        # Nettoyage des morceaux temporaires
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

def _trim_outer_silence(seg: AudioSegment, *, min_silence_len_ms: int = 50) -> AudioSegment:
    """Supprime le silence au début/à la fin d'un segment TTS (utile pour coller au timestamp)."""
    try:
        if seg.dBFS == float("-inf"):
            return seg
    except Exception:
        return seg

    # Seuil dynamique plus bas pour détecter plus de silences
    silence_thresh = min(-35.0, seg.dBFS - 14.0)
    try:
        ranges = detect_nonsilent(
            seg,
            min_silence_len=min_silence_len_ms,
            silence_thresh=silence_thresh,
        )
    except Exception:
        return seg

    if not ranges:
        return seg

    start_ms = max(0, int(ranges[0][0]))
    end_ms = min(len(seg), int(ranges[-1][1]))
    if end_ms <= start_ms:
        return seg

    return seg[start_ms:end_ms]

async def generate_french_audio_from_subtitles(subtitle_path: str, output_path: str):
    """Génère l'audio Français en respectant les timings de chaque ligne de sous-titres.

    - Utilise les sous-titres FR (déjà traduits) avec leurs timestamps,
    - Génère un segment audio par bloc,
    - Vitesse de lecture naturelle (aucun time-stretch ni ajustement de durée),
    - Positionne chaque segment au bon instant dans une piste audio globale.
    """
    print("4. Génération de l'audio Français (Edge-TTS) aligné sur les sous-titres...")
    voice = EDGE_TTS_VOICE
    rate = EDGE_TTS_RATE

    cues = _parse_subtitles_with_times(subtitle_path)
    if not cues:
        raise ValueError("Aucun bloc de sous-titres trouvé pour la synthèse vocale.")

    # Durée totale = fin du dernier bloc + marge pour segments à durée naturelle
    last_end_ms = max(end_ms for _, end_ms, _ in cues)
    full_duration_ms = int(last_end_ms + 5000)

    # Edge-TTS renvoie typiquement du MP3 24kHz mono. Fixer un format cible évite des surprises lors des overlays.
    target_frame_rate = 24000
    target_channels = 1

    full = AudioSegment.silent(duration=full_duration_ms, frame_rate=target_frame_rate).set_channels(target_channels)

    temp_files = []
    try:
        for idx, (start_ms, end_ms, text) in enumerate(cues):
            temp_path = f"{output_path}.cue{idx}.mp3"
            temp_files.append(temp_path)
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(temp_path)

            seg = AudioSegment.from_file(temp_path, format="mp3")
            seg = seg.set_frame_rate(target_frame_rate).set_channels(target_channels)
            seg = _trim_outer_silence(seg)

            if len(seg) > 0:
                full = full.overlay(seg, position=int(start_ms))

        out_ext = os.path.splitext(output_path)[1].lower().lstrip(".") or "mp3"
        full.export(output_path, format=out_ext)
        print(f"   -> Audio aligné sur les sous-titres généré : {output_path}")
    finally:
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    pass
            except Exception:
                pass

def extract_voice_sample(video_path: str, output_wav_path: str, duration_seconds: float = 12) -> str:
    """Extrait un court segment d'audio du début de la vidéo pour servir d'échantillon vocal (clonage).
    Retourne le chemin du fichier WAV créé. Utilise la vidéo avec piste audio pour cette étape uniquement.
    """
    print(f"   Extraction d'un échantillon vocal ({duration_seconds:.0f} s) depuis la vidéo...")
    video = None
    audio_clip = None
    try:
        video = VideoFileClip(video_path, audio=True)
        if video.audio is None:
            print("   -> Avertissement : pas de piste audio dans la vidéo, échantillon non créé.")
            return ""
        duration = min(float(duration_seconds), float(video.audio.duration))
        if duration < 2:
            print("   -> Avertissement : piste audio trop courte pour un échantillon.")
            return ""
        
        # Utiliser la méthode correcte pour AudioFileClip - utiliser slice au lieu de subclip
        audio_clip = video.audio[:duration]  # Utilisation du slicing Python
        audio_clip.write_audiofile(output_wav_path, fps=22050, nbytes=2, codec="pcm_s16le", logger=None)
        print(f"   -> Échantillon vocal enregistré : {output_wav_path}")
        return output_wav_path
    except Exception as e:
        print(f"   -> Impossible d'extraire l'échantillon vocal : {e}")
        return ""
    finally:
        for c in (audio_clip, video):
            try:
                if c is not None:
                    c.close()
            except Exception:
                pass


def _elevenlabs_create_voice(api_key: str, wav_path: str, name: str = "dub_source") -> str | None:
    """Crée une voix Instant Voice Clone ElevenLabs à partir d'un fichier WAV. Retourne voice_id ou None."""
    url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {"xi-api-key": api_key}
    try:
        with open(wav_path, "rb") as f:
            files = {"files": (os.path.basename(wav_path), f, "audio/wav")}
            data = {"name": name}
            r = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        r.raise_for_status()
        return r.json().get("voice_id")
    except Exception as e:
        print(f"   -> ElevenLabs création de voix échouée : {e}")
        return None


def generate_french_audio_from_subtitles_elevenlabs(
    api_key: str, voice_id: str, subtitle_path: str, output_path: str
) -> None:
    """Génère l'audio français avec la voix clonée ElevenLabs, aligné sur les timestamps des sous-titres."""
    print("4. Génération de l'audio Français (ElevenLabs, voix clonée) aligné sur les sous-titres...")
    cues = _parse_subtitles_with_times(subtitle_path)
    if not cues:
        raise ValueError("Aucun bloc de sous-titres trouvé pour la synthèse vocale.")
    last_end_ms = max(end_ms for _, end_ms, _ in cues)
    full_duration_ms = int(last_end_ms + 5000)
    target_frame_rate = 24000
    target_channels = 1
    full = AudioSegment.silent(duration=full_duration_ms, frame_rate=target_frame_rate).set_channels(target_channels)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {"text": "", "model_id": "eleven_multilingual_v2"}
    temp_files = []
    try:
        for idx, (start_ms, end_ms, text) in enumerate(cues):
            if not text.strip():
                continue
            temp_path = f"{output_path}.cue{idx}.mp3"
            temp_files.append(temp_path)
            payload["text"] = text
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(r.content)
            seg = AudioSegment.from_file(temp_path, format="mp3")
            seg = seg.set_frame_rate(target_frame_rate).set_channels(target_channels)
            seg = _trim_outer_silence(seg)
            if len(seg) > 0:
                full = full.overlay(seg, position=int(start_ms))
        out_ext = os.path.splitext(output_path)[1].lower().lstrip(".") or "mp3"
        full.export(output_path, format=out_ext)
        print(f"   -> Audio aligné (voix clonée) généré : {output_path}")
    finally:
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass


def merge_audio_video(video_path, new_audio_path, output_path):
    """Assemble la vidéo originale avec le nouvel audio (remplace l'audio original, pas de mix)."""
    print("5. Assemblage final de la vidéo...")
    video = None
    new_audio = None
    final_video = None
    try:
        # Charger la vidéo SANS l'audio original pour éviter toute superposition de voix
        video = VideoFileClip(video_path, audio=False)
        new_audio = AudioFileClip(new_audio_path)

        # Informations de durée
        try:
            v_dur = float(video.duration)
        except Exception:
            v_dur = None
        try:
            a_dur = float(new_audio.duration)
        except Exception:
            a_dur = None

        if v_dur is not None and a_dur is not None:
            print(f"   -> Durée vidéo : {v_dur:0.2f} s, durée audio : {a_dur:0.2f} s")

        # On NE MODIFIE PAS la durée de la vidéo.
        # Si l'audio est plus long que la vidéo, on le tronque pour qu'il ait exactement la même durée.
        if v_dur is not None and a_dur is not None and a_dur > v_dur + 0.05:
            print("   -> L'audio est plus long que la vidéo, il sera tronqué pour correspondre à la durée vidéo.")
            try:
                new_audio = new_audio.subclip(0, v_dur)
            except Exception:
                pass

        # moviepy v2 uses `with_audio` to attach an audio clip
        final_video = video.with_audio(new_audio)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)

        print("   -> Vidéo finale sauvegardée.")
    except Exception as e:
        print(f"Erreur lors de l'assemblage : {e}")
        raise
    finally:
        # S'assurer que tous les clips sont fermés pour libérer les fichiers
        for clip in (final_video, video, new_audio):
            try:
                if clip is not None:
                    clip.close()
            except Exception:
                pass

def process_directory_batch(input_dir: str, output_dir: str, cleanup: bool = True):
    """Traite tous les fichiers vidéo d'un répertoire et leurs sous-titres correspondants.
    
    Args:
        input_dir: Répertoire source contenant vidéos et sous-titres
        output_dir: Répertoire de sortie pour les vidéos doublées
        cleanup: Supprimer les fichiers temporaires après traitement
    """
    if not os.path.exists(input_dir):
        print(f"Erreur : Le répertoire d'entrée '{input_dir}' n'existe pas.")
        return
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Traitement du répertoire : {input_dir}")
    print(f"📂 Sortie vers : {output_dir}")
    print("=" * 60)
    
    # Trouver tous les fichiers vidéo récursivement dans le répertoire et sous-répertoires
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    input_path = Path(input_dir)
    for ext in video_extensions:
        # Recherche récursive dans tous les sous-dossiers
        video_files.extend(input_path.rglob(f"*{ext}"))
        video_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    if not video_files:
        print("Aucun fichier vidéo trouvé dans le répertoire et ses sous-répertoires.")
        print("Extensions recherchées :", ", ".join(video_extensions))
        return
    
    print(f"🎬 {len(video_files)} vidéo(s) trouvée(s) dans l'arborescence")
    print()
    
    processed_count = 0
    failed_count = 0
    
    for video_file in video_files:
        video_path = str(video_file)
        video_name = video_file.stem
        
        print(f"🎥 Traitement de : {video_file.relative_to(input_path)}")
        
        # Chercher les sous-titres correspondants dans le même répertoire que la vidéo
        subtitle_path = None
        subtitle_patterns = [
            f"{video_name}.srt",
            f"{video_name}.vtt", 
            f"{video_name}.en_US.vtt",
            f"{video_name}.en.vtt"
        ]
        
        for pattern in subtitle_patterns:
            potential_subtitle = video_file.parent / pattern
            if potential_subtitle.exists():
                subtitle_path = str(potential_subtitle)
                break
        
        if not subtitle_path:
            print(f"   ❌ Aucun sous-titre trouvé pour {video_file.name}")
            failed_count += 1
            print()
            continue
        
        # Créer la structure de sous-répertoires dans la sortie
        relative_path = video_file.relative_to(input_path)
        output_subdir = os.path.join(output_dir, str(relative_path.parent))
        os.makedirs(output_subdir, exist_ok=True)
        
        # Chemin de sortie pour la vidéo doublée
        output_video = os.path.join(output_subdir, f"{video_name}_fr.mp4")
        
        try:
            # Traiter cette vidéo
            run_process(video_path, subtitle_path, output_video, cleanup)
            processed_count += 1
            print(f"   ✅ Succès : {video_file.name} -> {os.path.basename(output_video)}")
        except Exception as e:
            print(f"   ❌ Erreur lors du traitement de {video_file.name} : {e}")
            failed_count += 1
        
        print()
    
    print("=" * 60)
    print(f"📊 Résumé du traitement :")
    print(f"   ✅ Réussies : {processed_count}")
    print(f"   ❌ Échouées : {failed_count}")
    print(f"   📁 Total : {processed_count + failed_count}")
    print(f"📂 Videos doublées disponibles dans : {output_dir}")


def clean_up():
    """Nettoie les fichiers temporaires."""
    files = [TEMP_AUDIO_FR]
    for f in files:
        if os.path.exists(f):
            os.remove(f)


class TextRedirector(object):
    """Redirect stdout/stderr to a tkinter Text widget (and original stream)."""
    def __init__(self, text_widget, stream):
        self.text_widget = text_widget
        self.stream = stream

    def write(self, message):
        try:
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')
        except Exception:
            pass
        try:
            self.stream.write(message)
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass


def run_process(input_video, input_subtitle, output_video, cleanup=True, output_subtitle_path=None):
    """Exécute tout le pipeline de doublage à partir de sous-titres déjà existants.

    This function is safe to run in a background thread.
    """
    timings = {}
    t_total0 = time.perf_counter()
    try:
        if not input_video or not input_subtitle:
            print("Erreur : vidéo source ou sous-titres non spécifiés.")
            return
        if not os.path.exists(input_video):
            print(f"Erreur : Le fichier vidéo '{input_video}' est introuvable.")
            return
        if not os.path.exists(input_subtitle):
            print(f"Erreur : Le fichier de sous-titres '{input_subtitle}' est introuvable.")
            return

        # Extraire un court échantillon vocal de la vidéo (pour clonage éventuel)
        voice_sample_path = extract_voice_sample(
            input_video,
            os.path.abspath(VOICE_SAMPLE_FILENAME),
            VOICE_SAMPLE_DURATION_S,
        )

        # Normalisation du chemin de sortie vidéo
        if not output_video:
            base, _ = os.path.splitext(input_video)
            output_video = base + "_fr.mp4"
            print(f"Avertissement : aucun fichier vidéo de sortie spécifié, utilisation de : {output_video}")
        else:
            base, ext = os.path.splitext(output_video)
            if not ext:
                output_video = output_video + ".mp4"

        # Génération du chemin de sortie des sous-titres FR (même extension, suffixe _fr)
        if output_subtitle_path:
            output_subtitle = output_subtitle_path
        else:
            root, ext = os.path.splitext(input_subtitle)
            output_subtitle = root + "_fr" + (ext or ".vtt")

        t0 = time.perf_counter()
        english_subtitles = _read_subtitle_text(input_subtitle)
        french_subtitles = translate_with_llama(english_subtitles, output_subtitle)
        timings["translate_s"] = time.perf_counter() - t0

        # Génération audio : si clé ElevenLabs + échantillon vocal, on clone la voix source ; sinon Edge-TTS.
        t0 = time.perf_counter()
        if ELEVEN_LABS_API_KEY and voice_sample_path and os.path.exists(voice_sample_path):
            voice_id = _elevenlabs_create_voice(ELEVEN_LABS_API_KEY, voice_sample_path)
            if voice_id:
                generate_french_audio_from_subtitles_elevenlabs(
                    ELEVEN_LABS_API_KEY, voice_id, output_subtitle, TEMP_AUDIO_FR
                )
            else:
                asyncio.run(generate_french_audio_from_subtitles(output_subtitle, TEMP_AUDIO_FR))
        else:
            asyncio.run(generate_french_audio_from_subtitles(output_subtitle, TEMP_AUDIO_FR))
        timings["tts_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        merge_audio_video(input_video, TEMP_AUDIO_FR, output_video)
        timings["merge_s"] = time.perf_counter() - t0

        print(f"\n✅ SUCCÈS ! La vidéo doublée est '{output_video}'")
        print(f"   -> Sous-titres français : {output_subtitle}")
    except Exception as e:
        print(f"\n❌ Le processus a échoué : {e}")
    finally:
        timings["total_s"] = time.perf_counter() - t_total0
        if cleanup:
            t0 = time.perf_counter()
            clean_up()
            timings["cleanup_s"] = time.perf_counter() - t0
        _print_timing_report(timings)


def start_in_thread(btn_start, in_path_var, sub_path, sub_out_path_var, out_path_var, cleanup_var, log_widget):
    """Callback to start the dubbing process in a new thread and manage UI state.
    sub_path est le chemin des sous-titres (dérivé du nom de la vidéo, .srt ou .vtt).
    """
    def target():
        try:
            out_sub = sub_out_path_var.get().strip() or None
            run_process(in_path_var.get(), sub_path, out_path_var.get(), cleanup_var.get(), output_subtitle_path=out_sub)
        finally:
            # Re-enable start button in main thread
            btn_start.configure(state='normal')

    # Disable the start button while running
    btn_start.configure(state='disabled')
    t = threading.Thread(target=target, daemon=True)
    t.start()


def build_and_launch_gui():
    if tk is None:
        print("Tkinter n'est pas disponible dans cet environnement. Impossible de lancer l'interface graphique.")
        return

    root = tk.Tk()
    root.title("Dub3 - Doublage depuis sous-titres")
    root.geometry("800x520")

    frm = tk.Frame(root)
    frm.pack(fill='both', expand=True, padx=8, pady=8)

    # Input selection (vidéo par défaut : output.mp4 dans le répertoire courant)
    in_path_var = tk.StringVar(value=os.path.abspath("output.mp4"))
    sub_path_var = tk.StringVar(value="")
    sub_out_path_var = tk.StringVar(value="")
    out_path_var = tk.StringVar(value="")
    cleanup_var = tk.BooleanVar(value=True)

    def browse_input():
        p = filedialog.askopenfilename(title='Sélectionner la vidéo source', filetypes=[('Video files', '*.mp4;*.mov;*.mkv'), ('All files','*.*')])
        if p:
            in_path_var.set(p)

    def browse_subtitle_output():
        p = filedialog.asksaveasfilename(
            title='Enregistrer les sous-titres français sous',
            defaultextension='.srt',
            filetypes=[('Sous-titres', '*.srt;*.vtt;*.ass;*.ssa'), ('Tous les fichiers','*.*')]
        )
        if p:
            sub_out_path_var.set(p)

    def browse_output():
        p = filedialog.asksaveasfilename(title='Enregistrer la vidéo de sortie sous', defaultextension='.mp4', filetypes=[('MP4','*.mp4')])
        if p:
            out_path_var.set(p)

    row = tk.Frame(frm)
    tk.Label(row, text='Fichier vidéo source:').pack(side='left')
    tk.Entry(row, textvariable=in_path_var, width=60).pack(side='left', padx=6)
    tk.Button(row, text='Parcourir', command=browse_input).pack(side='left')
    row.pack(fill='x', pady=6)

    # Sous-titres anglais : même nom que la vidéo, extension .srt ou .vtt (pas de sélection manuelle)
    row_sub_out = tk.Frame(frm)
    tk.Label(row_sub_out, text='Sous-titres français (sortie):').pack(side='left')
    tk.Entry(row_sub_out, textvariable=sub_out_path_var, width=60).pack(side='left', padx=6)
    tk.Button(row_sub_out, text='Choisir...', command=browse_subtitle_output).pack(side='left')
    row_sub_out.pack(fill='x', pady=6)

    row2 = tk.Frame(frm)
    tk.Label(row2, text='Fichier vidéo de sortie:').pack(side='left')
    tk.Entry(row2, textvariable=out_path_var, width=60).pack(side='left', padx=6)
    tk.Button(row2, text='Choisir...', command=browse_output).pack(side='left')
    row2.pack(fill='x', pady=6)

    options_row = tk.Frame(frm)
    tk.Checkbutton(options_row, text='Supprimer les fichiers temporaires à la fin', variable=cleanup_var).pack(side='left')
    options_row.pack(fill='x', pady=6)

    # Start button and simple instructions
    btn_row = tk.Frame(frm)
    btn_start = tk.Button(btn_row, text='Démarrer le doublage', width=20, bg='#4CAF50', fg='white')
    btn_start.pack(side='left', padx=6)
    btn_row.pack(fill='x', pady=6)

    # Log area
    log_label = tk.Label(frm, text='Journal')
    log_label.pack(anchor='w')
    log_widget = scrolledtext.ScrolledText(frm, height=18, state='disabled')
    log_widget.pack(fill='both', expand=True)

    # Redirect stdout/stderr to the log widget while process runs
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def on_start():
        # Vérifier que le fichier vidéo est renseigné
        video_path = in_path_var.get().strip()
        if not video_path:
            messagebox.showerror("Chemin manquant", "Veuillez sélectionner un fichier vidéo source.")
            return
        # Sous-titres : même nom que la vidéo, extension .srt ou .vtt
        sub_path = _subtitle_path_from_video(video_path)
        if not sub_path:
            messagebox.showerror(
                "Sous-titres introuvables",
                "Aucun fichier de sous-titres (.srt ou .vtt) portant le même nom que la vidéo n'a été trouvé dans le même répertoire."
            )
            return
        if not out_path_var.get().strip():
            # Proposer automatiquement un nom de sortie basé sur la vidéo
            base, _ = os.path.splitext(video_path)
            suggested = base + "_fr.mp4"
            out_path_var.set(suggested)

        # redirect
        sys.stdout = TextRedirector(log_widget, original_stdout)
        sys.stderr = TextRedirector(log_widget, original_stderr)
        start_in_thread(btn_start, in_path_var, sub_path, sub_out_path_var, out_path_var, cleanup_var, log_widget)

    def on_close():
        # restore streams
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        except Exception:
            pass
        root.destroy()

    btn_start.configure(command=on_start)
    root.protocol('WM_DELETE_WINDOW', on_close)

    # Helpful note
    note = tk.Label(frm, text="Astuce: si vous préférez la ligne de commande, exécutez ce script depuis un terminal."
                    , fg='gray')
    note.pack(anchor='w', pady=4)

    root.mainloop()

# --- EXÉCUTION ---
if __name__ == "__main__":
    # Provide a simple CLI so the requested procedure (open VTT, translate, TTS per-cue, assemble audio, dub video)
    # can be run from a terminal. If no CLI args are provided the GUI is launched (if available).
    parser = argparse.ArgumentParser(description="Doublage vidéo depuis un fichier de sous-titres (VTT/SRT)")
    parser.add_argument('--cli', action='store_true', help='Forcer le mode CLI')
    parser.add_argument('-v', '--video', help='Fichier vidéo source (ex: input.mp4)')
    parser.add_argument('-s', '--subtitle', help='Fichier de sous-titres source (VTT/SRT)')
    parser.add_argument('-o', '--output', help='Fichier vidéo de sortie (ex: output_fr.mp4)')
    parser.add_argument('--subtitle-out', help='Chemin de sortie pour les sous-titres traduits (optionnel)')
    parser.add_argument('--translate-first-only', action='store_true', help='Traduire uniquement le premier bloc de sous-titres et sortir')
    parser.add_argument('--no-cleanup', action='store_true', help="Ne pas supprimer les fichiers temporaires à la fin")
    parser.add_argument('--batch-dir', help='Répertoire contenant les vidéos et sous-titres à traiter en lot')
    parser.add_argument('--batch-output', help='Répertoire de sortie pour les vidéos doublées (défaut: input_dir_dubbed)')

    args = parser.parse_args()

    # Traitement par lot (prioritaire sur les autres options)
    if args.batch_dir:
        if not os.path.exists(args.batch_dir):
            print(f"Erreur : Le répertoire '{args.batch_dir}' n'existe pas.")
            sys.exit(2)
        
        output_dir = args.batch_output
        if not output_dir:
            # Créer automatiquement un nom de répertoire de sortie
            input_path = Path(args.batch_dir)
            output_dir = str(input_path.parent / f"{input_path.name}_dubbed")
        
        cleanup = not args.no_cleanup
        process_directory_batch(args.batch_dir, output_dir, cleanup)
        sys.exit(0)

    # Dérivation des sous-titres depuis la vidéo si non fournis (même nom, .srt ou .vtt)
    if args.video and not args.subtitle:
        args.subtitle = _subtitle_path_from_video(args.video)
        if not args.subtitle:
            print(f"Erreur: aucun fichier de sous-titres (.srt ou .vtt) trouvé pour la vidéo '{args.video}'")
            sys.exit(2)

    # If user requested translate-first-only, handle it and exit early
    if args.translate_first_only:
        if not args.subtitle:
            print("Erreur: --translate-first-only nécessite -v (vidéo) ou -s / --subtitle pour le fichier VTT/SRT")
            sys.exit(2)
        out_sub = args.subtitle_out
        if not out_sub:
            root, ext = os.path.splitext(args.subtitle)
            out_sub = root + "_first_fr" + (ext or ".vtt")
        translate_first_cue(args.subtitle, out_sub)
        sys.exit(0)

    # En mode CLI sans -v, utiliser output.mp4 dans le répertoire courant
    if args.cli and not args.video:
        args.video = "output.mp4"
        args.subtitle = _subtitle_path_from_video(args.video)
        if not args.subtitle:
            print("Erreur: aucun fichier de sous-titres (output.srt ou output.vtt) trouvé dans le répertoire courant.")
            sys.exit(2)

    # If user asked CLI or provided video (with auto subtitle), run without GUI
    if args.cli or (args.video and args.subtitle):
        cleanup = not args.no_cleanup
        run_process(args.video, args.subtitle, args.output, cleanup, output_subtitle_path=args.subtitle_out)
    else:
        build_and_launch_gui()
