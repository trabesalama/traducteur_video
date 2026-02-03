import os
import asyncio
from dotenv import load_dotenv
from groq import Groq
import edge_tts
from moviepy import VideoFileClip, AudioFileClip

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

# --- EXÉCUTION ---
if __name__ == "__main__":
    # Vérification que le fichier vidéo existe
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
            # Décommentez la ligne ci-dessous pour supprimer les fichiers temporaires automatiquement
            clean_up()