import os
import asyncio
import time
import re
from dotenv import load_dotenv
from groq import Groq
import edge_tts
from moviepy import VideoFileClip, AudioFileClip
from pydub import AudioSegment

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

# 2. Configuration des fichiers (valeurs par défaut, modifiables via l'UI ou la CLI)
# Chemins par défaut fournis par l'utilisateur
INPUT_VIDEO = r"D:/tah/D4 - AI AGENTIC/Udemy - The Complete Agentic AI Engineering Course (2025) 2025-4/1 - Week 1/15 -Day 3 - Connecting Agentic Patterns to Tool Use Essential AI Building Blocks.mp4"
INPUT_SUBTITLE = r"D:/tah/D4 - AI AGENTIC/Udemy - The Complete Agentic AI Engineering Course (2025) 2025-4/1 - Week 1/15 -Day 3 - Connecting Agentic Patterns to Tool Use Essential AI Building Blocks.vtt"
FRENCH_SUBTITLE_DEFAULT = r"D:/tah/D4 - AI AGENTIC/Udemy - The Complete Agentic AI Engineering Course (2025) 2025-4/1 - Week 1/15 -Day 3 - Connecting Agentic Patterns to Tool Use Essential AI Building Blocks_fr.vtt"
OUTPUT_VIDEO = r"D:/tah/D4 - AI AGENTIC/Udemy - The Complete Agentic AI Engineering Course (2025) 2025-4/1 - Week 1/15 -Day 3 - Connecting Agentic Patterns to Tool Use Essential AI Building Blocks_fr.mp4"
TEMP_AUDIO_FR = "temp_audio_fr.mp3"    # Audio français temporaire (temporaire dans le projet)

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

def _read_subtitle_text(path: str) -> str:
    """Lit le fichier de sous-titres tel quel (SRT, VTT, etc.)."""
    print(f"1. Chargement des sous-titres : {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier de sous-titres '{path}' est introuvable.")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    print("   -> Sous-titres chargés.")
    return content

def _parse_time_to_seconds(time_str: str) -> float:
    """Convertit un timestamp SRT/VTT en secondes (ex: 00:01:02,500 ou 00:01:02.500)."""
    time_str = time_str.strip()
    # Remplace la virgule par un point pour uniformiser
    time_str = time_str.replace(",", ".")
    hms, _, ms = time_str.partition(".")
    parts = hms.split(":")
    if len(parts) != 3:
        raise ValueError(f"Format de temps invalide: {time_str}")
    h, m, s = parts
    base = int(h) * 3600 + int(m) * 60 + int(s)
    frac = float("0." + ms) if ms else 0.0
    return base + frac

def _parse_subtitles_with_times(path: str):
    """Parse un fichier SRT/VTT et renvoie une liste de (start_s, end_s, text)."""
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
                start_s = _parse_time_to_seconds(left.strip())
                end_s = _parse_time_to_seconds(right.strip().split(" ")[0])
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
                cues.append((start_s, end_s, text))
        else:
            i += 1

    print(f"   -> {len(cues)} blocs de sous-titres détectés.")
    return cues

def translate_with_llama(text: str, output_subtitle_path: str) -> str:
    """Traduit le contenu des sous-titres via Groq en conservant le format.

    On demande explicitement de retourner des sous-titres au même format (SRT/VTT, etc.).
    """
    print("2. Traduction des sous-titres avec Groq (Llama 4 Scout)...")
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    Tu es un expert en sous-titres et en doublage.
    Le texte ci-dessous est un fichier de sous-titres en anglais (par exemple SRT ou VTT).
    Traduis UNIQUEMENT le texte visible à l'écran en français, en respectant STRICTEMENT les règles suivantes :
    - tu dois conserver EXACTEMENT toutes les lignes contenant "-->" (les timestamps) SANS les modifier,
    - tu ne changes PAS les numéros de blocs s'il y en a,
    - tu ne modifies JAMAIS le format des timestamps ni leur position,
    - SRT doit rester SRT, VTT doit rester VTT,
    - tu remplaces simplement le texte dialogué par sa traduction française sur les lignes prévues pour le texte.
    
    Fichier de sous-titres :
    {text}
    
    Retourne UNIQUEMENT le fichier de sous-titres complet en français, au même format, avec les mêmes timestamps.
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
    - Si une ligne est encore trop longue, on la re-découpe par phrases.
    """
    def _split_long_line(line: str, limit: int):
        # Découpe grossière par ponctuation forte
        sentences = re.split(r'(?<=[\.\?\!])\s+', line)
        parts = []
        current = []
        current_len = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            extra = len(s) + 1
            if current_len + extra > limit and current:
                parts.append(" ".join(current))
                current = [s]
                current_len = len(s)
            else:
                current.append(s)
                current_len += extra
        if current:
            parts.append(" ".join(current))
        return parts

    chunks = []
    current = []
    current_len = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Si la ligne dépasse la limite à elle seule, on la découpe en sous-parties
        sub_lines = _split_long_line(line, max_chars) if len(line) > max_chars else [line]

        for sub in sub_lines:
            extra = len(sub) + 1  # +1 pour un saut de ligne / espace
            if current_len + extra > max_chars and current:
                chunks.append("\n".join(current))
                current = [sub]
                current_len = len(sub)
            else:
                current.append(sub)
                current_len += extra

    if current:
        chunks.append("\n".join(current))

    print(f"   -> Texte découpé en {len(chunks)} morceau(x) pour Edge-TTS.")
    return chunks

async def generate_french_audio(text, output_path):
    """Génère l'audio Français via Edge-TTS (avec découpe en chunks pour ne rien couper)."""
    print("4. Génération de l'audio Français (Edge-TTS)...")
    voice = "fr-FR-DeniseNeural"

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
            communicate = edge_tts.Communicate(chunk, voice)
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

async def generate_french_audio_from_subtitles(subtitle_path: str, output_path: str):
    """Génère l'audio Français en respectant les timings de chaque ligne de sous-titres.

    - Utilise les sous-titres FR (déjà traduits) avec leurs timestamps,
    - Génère un segment audio par bloc,
    - Adapte la vitesse/durée pour coller à la fenêtre temporelle,
    - Positionne chaque segment au bon instant dans une piste audio globale.
    """
    print("4. Génération de l'audio Français (Edge-TTS) aligné sur les sous-titres...")
    voice = "fr-FR-DeniseNeural"

    cues = _parse_subtitles_with_times(subtitle_path)
    if not cues:
        raise ValueError("Aucun bloc de sous-titres trouvé pour la synthèse vocale.")

    # Durée totale = fin du dernier bloc + petite marge
    last_end = max(end for _, end, _ in cues)
    full = AudioSegment.silent(duration=int((last_end + 0.5) * 1000))

    temp_files = []
    try:
        for idx, (start_s, end_s, text) in enumerate(cues):
            desired_ms = int(max(0.1, end_s - start_s) * 1000)
            if desired_ms <= 0:
                continue

            temp_path = f"{output_path}.cue{idx}.mp3"
            temp_files.append(temp_path)

            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(temp_path)

            seg = AudioSegment.from_file(temp_path, format="mp3")
            current_ms = len(seg)

            # Ajustement de durée : accélération/ralentissement + pad/trim
            if current_ms == 0:
                seg = AudioSegment.silent(duration=desired_ms)
            else:
                speed = current_ms / desired_ms
                new_frame_rate = int(seg.frame_rate * speed)
                stretched = seg._spawn(seg.raw_data, overrides={"frame_rate": new_frame_rate})
                stretched = stretched.set_frame_rate(seg.frame_rate)
                if len(stretched) > desired_ms:
                    seg = stretched[:desired_ms]
                else:
                    seg = stretched + AudioSegment.silent(duration=desired_ms - len(stretched))

            position_ms = int(start_s * 1000)
            full = full.overlay(seg, position=position_ms)

        full.export(output_path, format="mp3")
        print(f"   -> Audio aligné sur les sous-titres généré : {output_path}")
    finally:
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

def merge_audio_video(video_path, new_audio_path, output_path):
    """Assemble la vidéo originale avec le nouvel audio."""
    print("5. Assemblage final de la vidéo...")
    video = None
    new_audio = None
    final_video = None
    try:
        video = VideoFileClip(video_path)
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
        if not os.path.exists(input_video):
            print(f"Erreur : Le fichier '{input_video}' est introuvable.")
            return

        # Génération du chemin de sortie des sous-titres FR (même extension, suffixe _fr)
        if output_subtitle_path:
            output_subtitle = output_subtitle_path
        else:
            root, ext = os.path.splitext(input_subtitle)
            output_subtitle = root + "_fr" + ext

        t0 = time.perf_counter()
        english_subtitles = _read_subtitle_text(input_subtitle)
        french_subtitles = translate_with_llama(english_subtitles, output_subtitle)
        timings["translate_s"] = time.perf_counter() - t0

        # Pour la voix-off, on génère l'audio directement à partir des sous-titres FR,
        # en respectant les timestamps de chaque bloc.
        t0 = time.perf_counter()
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


def start_in_thread(btn_start, in_path_var, sub_path_var, sub_out_path_var, out_path_var, cleanup_var, log_widget):
    """Callback to start the dubbing process in a new thread and manage UI state."""
    def target():
        try:
            # Ici, in_path_var = vidéo, sub_path_var = sous-titres
            out_sub = sub_out_path_var.get().strip() or None
            run_process(in_path_var.get(), sub_path_var.get(), out_path_var.get(), cleanup_var.get(), output_subtitle_path=out_sub)
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

    # Input selection
    in_path_var = tk.StringVar(value=INPUT_VIDEO)
    sub_path_var = tk.StringVar(value=INPUT_SUBTITLE)
    sub_out_path_var = tk.StringVar(value="")
    out_path_var = tk.StringVar(value=OUTPUT_VIDEO)
    cleanup_var = tk.BooleanVar(value=True)

    def browse_input():
        p = filedialog.askopenfilename(title='Sélectionner la vidéo source', filetypes=[('Video files', '*.mp4;*.mov;*.mkv'), ('All files','*.*')])
        if p:
            in_path_var.set(p)

    def browse_subtitle():
        p = filedialog.askopenfilename(title='Sélectionner le fichier de sous-titres anglais', filetypes=[('Sous-titres', '*.srt;*.vtt;*.ass;*.ssa'), ('Tous les fichiers','*.*')])
        if p:
            sub_path_var.set(p)

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

    row_sub = tk.Frame(frm)
    tk.Label(row_sub, text='Sous-titres anglais:').pack(side='left')
    tk.Entry(row_sub, textvariable=sub_path_var, width=60).pack(side='left', padx=6)
    tk.Button(row_sub, text='Parcourir', command=browse_subtitle).pack(side='left')
    row_sub.pack(fill='x', pady=6)

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
        # redirect
        sys.stdout = TextRedirector(log_widget, original_stdout)
        sys.stderr = TextRedirector(log_widget, original_stderr)
        start_in_thread(btn_start, in_path_var, sub_path_var, sub_out_path_var, out_path_var, cleanup_var, log_widget)

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
    # If the user passes --cli, keep original CLI behavior. Otherwise launch the GUI if available.
    if '--cli' in sys.argv:
        # Mode CLI simple : utilise les constantes INPUT_VIDEO / INPUT_SUBTITLE / OUTPUT_VIDEO
        if not os.path.exists(INPUT_VIDEO):
            print(f"Erreur : Le fichier vidéo '{INPUT_VIDEO}' est introuvable.")
        elif not os.path.exists(INPUT_SUBTITLE):
            print(f"Erreur : Le fichier de sous-titres '{INPUT_SUBTITLE}' est introuvable.")
        else:
            try:
                run_process(INPUT_VIDEO, INPUT_SUBTITLE, OUTPUT_VIDEO, cleanup=True, output_subtitle_path=FRENCH_SUBTITLE_DEFAULT)
                print(f"\n✅ SUCCÈS ! La vidéo doublée est '{OUTPUT_VIDEO}'")
                print(f"   -> Sous-titres français : '{FRENCH_SUBTITLE_DEFAULT}'")
            except Exception as e:
                print(f"\n❌ Le processus a échoué : {e}")
            finally:
                clean_up()
    else:
        # Launch GUI (preferred)
        build_and_launch_gui()