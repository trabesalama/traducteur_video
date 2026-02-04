import os
import asyncio
import time
from dotenv import load_dotenv
from groq import Groq
import edge_tts
from moviepy import VideoFileClip, AudioFileClip

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

# 2. Configuration des fichiers
INPUT_VIDEO = "video.mp4"  # Vidéo dans le répertoire courant
OUTPUT_VIDEO = "video_output_french.mp4"
TEMP_AUDIO_EN = "temp_audio_en.wav"
TEXT_OUTPUT = "transcript.txt"
FRENCH_TEXT = "french_translation.txt"
TEMP_AUDIO_FR = "temp_audio_fr.mp3"

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
        ("extract_s", "Extraction audio"),
        ("transcribe_s", "Transcription"),
        ("translate_s", "Traduction"),
        ("tts_s", "Génération TTS"),
        ("merge_s", "Assemblage vidéo"),
        ("cleanup_s", "Nettoyage"),
    ):
        if key in timings:
            print(f"   - {label} : {_format_duration(timings[key])}")

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
    """Transcrit l'audio localement (faster-whisper, puis openai-whisper en fallback)."""
    import traceback
    print("2. Transcription locale (faster-whisper)...")

    # 1) faster_whisper
    try:
        from faster_whisper import WhisperModel
        print("   -> Utilisation de faster_whisper")
        model = WhisperModel("small", device="cpu")
        segments, _ = model.transcribe(audio_path)
        transcript = "\n".join([s.text for s in segments])

        with open(TEXT_OUTPUT, "w", encoding="utf-8") as f:
            f.write(transcript)
        print("   -> Transcription (faster_whisper) terminée.")
        return transcript
    except Exception as e_fw:
        print(f"   -> faster_whisper indisponible ou erreur: {e_fw}")

    # 2) openai-whisper fallback
    try:
        import whisper
        print("   -> Utilisation de openai-whisper en fallback")
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        transcript = result.get("text", "")

        with open(TEXT_OUTPUT, "w", encoding="utf-8") as f:
            f.write(transcript)
        print("   -> Transcription (openai-whisper) terminée.")
        return transcript
    except Exception as e_w:
        print(f"   -> Tous les moteurs de transcription ont échoué: {e_w}")
        traceback.print_exc()
        raise

def translate_with_llama(text):
    """Traduit le texte via Llama 4 Scout (modèle depuis votre image)."""
    print("3. Traduction avec Groq (Llama 4 Scout)...")
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    Tu es un expert en doublage. Traduis le texte suivant de l'anglais vers le français.
    Adapte le ton et essaie de garder une longueur de phrase compatible avec le timing vidéo.
    
    Texte :
    {text}
    
    Retourne UNIQUEMENT la traduction.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-4-scout", # Modèle spécifique de votre capture
            temperature=0.3,
        )

        french_text = chat_completion.choices[0].message.content
        
        with open(FRENCH_TEXT, "w", encoding="utf-8") as f:
            f.write(french_text)
        print("   -> Traduction terminée.")
        return french_text
    except Exception as e:
        # Fallback si le modèle exact n'est pas encore dispo (au cas où)
        print(f"Attention : Erreur avec llama-4-scout ({e}). Tentative avec un modèle standard...")
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", # Alternative robuste
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content

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
    files = [TEMP_AUDIO_EN, TEXT_OUTPUT, FRENCH_TEXT, TEMP_AUDIO_FR]
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


def run_process(input_video, output_video, cleanup=True):
    """Run the full dubbing pipeline using the provided paths.

    This function is safe to run in a background thread.
    """
    timings = {}
    t_total0 = time.perf_counter()
    try:
        if not os.path.exists(input_video):
            print(f"Erreur : Le fichier '{input_video}' est introuvable.")
            return

        t0 = time.perf_counter()
        extract_audio(input_video, TEMP_AUDIO_EN)
        timings["extract_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        original_text = transcribe_with_whisper_cpp(TEMP_AUDIO_EN)
        timings["transcribe_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        translated_text = translate_with_llama(original_text)
        timings["translate_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        asyncio.run(generate_french_audio(translated_text, TEMP_AUDIO_FR))
        timings["tts_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        merge_audio_video(input_video, TEMP_AUDIO_FR, output_video)
        timings["merge_s"] = time.perf_counter() - t0

        print(f"\n✅ SUCCÈS ! La vidéo doublée est '{output_video}'")
    except Exception as e:
        print(f"\n❌ Le processus a échoué : {e}")
    finally:
        timings["total_s"] = time.perf_counter() - t_total0
        if cleanup:
            t0 = time.perf_counter()
            clean_up()
            timings["cleanup_s"] = time.perf_counter() - t0
        _print_timing_report(timings)


def start_in_thread(btn_start, in_path_var, out_path_var, cleanup_var, log_widget):
    """Callback to start the dubbing process in a new thread and manage UI state."""
    def target():
        try:
            run_process(in_path_var.get(), out_path_var.get(), cleanup_var.get())
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
    root.title("Dub2 - Interface de doublage")
    root.geometry("800x520")

    frm = tk.Frame(root)
    frm.pack(fill='both', expand=True, padx=8, pady=8)

    # Input selection
    in_path_var = tk.StringVar(value=INPUT_VIDEO)
    out_path_var = tk.StringVar(value=OUTPUT_VIDEO)
    cleanup_var = tk.BooleanVar(value=True)

    def browse_input():
        p = filedialog.askopenfilename(title='Sélectionner la vidéo source', filetypes=[('Video files', '*.mp4;*.mov;*.mkv'), ('All files','*.*')])
        if p:
            in_path_var.set(p)

    def browse_output():
        p = filedialog.asksaveasfilename(title='Enregistrer la vidéo de sortie sous', defaultextension='.mp4', filetypes=[('MP4','*.mp4')])
        if p:
            out_path_var.set(p)

    row = tk.Frame(frm)
    tk.Label(row, text='Fichier vidéo source:').pack(side='left')
    tk.Entry(row, textvariable=in_path_var, width=60).pack(side='left', padx=6)
    tk.Button(row, text='Parcourir', command=browse_input).pack(side='left')
    row.pack(fill='x', pady=6)

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
        start_in_thread(btn_start, in_path_var, out_path_var, cleanup_var, log_widget)

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
        # CLI path (original behavior)
        if not os.path.exists(INPUT_VIDEO):
            print(f"Erreur : Le fichier '{INPUT_VIDEO}' est introuvable dans le répertoire actuel.")
        else:
            try:
                extract_audio(INPUT_VIDEO, TEMP_AUDIO_EN)
                original_text = transcribe_with_whisper_cpp(TEMP_AUDIO_EN)
                translated_text = translate_with_llama(original_text)
                asyncio.run(generate_french_audio(translated_text, TEMP_AUDIO_FR))
                merge_audio_video(INPUT_VIDEO, TEMP_AUDIO_FR, OUTPUT_VIDEO)
                print("\n✅ SUCCÈS ! La vidéo doublée est 'video_output_french.mp4'")
            except Exception as e:
                print(f"\n❌ Le processus a échoué : {e}")
            finally:
                clean_up()
    else:
        # Launch GUI (preferred)
        build_and_launch_gui()