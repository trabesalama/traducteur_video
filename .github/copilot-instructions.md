# Copilot / AI Agent instructions for traducteur ✅

## Quick project summary
- Purpose: translate and dub videos (English → French), with both CLI and simple GUI flows.
- Major components:
  - Translation: `dub3.py`, `dub4.py`, `dub_video_from_srt.py` (uses Groq/OpenRouter → fallback Google)
  - Transcription: local-first (whisper_cpp → faster_whisper → openai-whisper) in `dub4.py`
  - TTS: `edge-tts` (primary) and Coqui `TTS` (used in smaller scripts). See `dub3.py`, `dub4.py`, `dub_video_from_srt.py`.
  - Media I/O & assembly: `ffmpeg` (external CLI), `moviepy`, `pydub`.
  - Helpers: subtitle parsing, time alignment, audio/video merging (see `dub3.py` / `dub4.py`).

## Key developer workflows ⚙️
- Activate environment:
  - Windows PowerShell: `& .\.venv\Scripts\Activate.ps1` or run `activate_venv.ps1`.
- CLI examples:
  - Full pipeline (translate subtitles, generate aligned audio, merge into video):
    - `python dub3.py --cli -v input.mp4 -s input.srt -o output_fr.mp4`
  - Generate single audio from translated subtitles:
    - `python generate_audio_from_subtitles.py -s subtitle_fr.srt -o out_audio.mp3`
  - Run the main advanced pipeline (concurrency / cache options):
    - `python dub4.py --concurrency 6 --block-concurrency 4 --cache-path translation_cache.json` 
- Quick tests:
  - `python test_tts_memory.py` (runs in-memory TTS routines)
  - `python test.py` (simple local SRT→TTS script usage)

## Environment & external dependencies ⚠️
- Required env vars (loaded via `.env` & `python-dotenv`):
  - `GROQ_API_KEY` (most translation flows require this) — enforced in `dub3.py` / `dub4.py`.
  - `OPENROUTER_API_KEY` (used in `dub_video_from_srt.py`).
  - Optional: `GROQ_MODELS`, `TRANSLATION_CACHE_PATH`, `AUDIO_CACHE_PATH`, `AUDIO_CACHE_TTL`.
- System tools: `ffmpeg` must be installed and on PATH for extraction/merging.
- GUI: `tkinter` is optional (used by `dub3.py` if available).

## Important project conventions and patterns 📌
- Subtitle safety: translation functions (e.g., `translate_with_llama` in `dub3.py`) must preserve timestamps and block structure exactly. Prompts explicitly instruct to not modify lines with `-->` timestamps.
- Multi-layer fallback approach is pervasive:
  - Translation: Groq (preferred models list) → Google translator (deep_translator) as fallback.
  - Transcription: `whisper_cpp` → `faster_whisper` → `openai-whisper`.
  - TTS: `edge-tts` primary, Coqui TTS used in smaller scripts.
- Caching: translations and generated audio are cached to avoid repeated external calls.
  - Translation cache file: `translation_cache.json` (TTL controlled via env/args).
  - Audio cache dir: `./audio_cache` (configurable).
  - Use provided helpers: `get_cached_translation`, `set_cached_translation`, `_read_cached_audio`, `_write_audio_cache`.
- Concurrency & safety:
  - TTS generation uses semaphores and per-key locks to avoid duplicate simultaneous work (`_get_audio_cache_lock` / `asyncio.BoundedSemaphore`).
  - Respect the CLI flags that control concurrency (e.g., `--concurrency`, `--block-concurrency`).

## Code patterns to follow when contributing ✍️
- Keep prompts deterministic and explicit when translating subtitles: ask the model to "return ONLY the translated subtitle file in the same format".
- When adding features that touch media files, prefer using existing helpers (`parse_srt`, `_parse_subtitles_with_times`, `generate_french_audio_from_subtitles`) to maintain alignment semantics.
- When changing defaults for caching or concurrency, expose them via CLI flags and environment variables so users can tune them without editing code.
- For TTS tuning, use provided `compute_rate_percent` and `shorten_text_to_fit` logic rather than ad-hoc duration adjustments.

## Testing & debugging tips 🐞
- Reproduce small pieces locally: use `test.py`, `test_tts_memory.py` for TTS; these show how TTS models are invoked.
- Inspect `translation_cache.json` and `audio_cache/` to confirm cache hits/misses and to debug duplication.
- For transcription fallbacks, ensure a machine has the requested local model (whisper_cpp or faster_whisper) installed; otherwise logs will indicate fallback path.

## Files to inspect for more context
- `dub3.py` — simple, GUI + CLI pipeline, Edge-TTS chunking and alignment (good for examples of subtitle handling).
- `dub4.py` — advanced pipeline: robust caching, concurrency, multiple fallbacks, and CLI flags.
- `dub_video_from_srt.py`, `dub_video_s2s.py` — other practical examples (Coqui TTS and OpenRouter usage).
- `generate_audio_from_subtitles.py` — CLI wrapper that uses `dub3` helpers.

---
## Suggested prompts & safety checks 📜

- Example prompt for SRT/VTT translation (suitable for `translate_with_llama` in `dub3.py`):

  "You are an expert in subtitles and dubbing. Translate ONLY the visible subtitle text from English to natural spoken French. DO NOT modify any lines that contain timestamps (lines with '-->'), DO NOT change block numbers or timestamp formats, and DO NOT add or remove any lines. Return ONLY the translated subtitle file in the same format (SRT/VTT), and nothing else."

- Example prompt for translating a single cue (used in `translate_first_cue`):

  "Translate only the following subtitle text to French. Return only the translated text (no explanations, no extra lines)."

- Quick safety checks agents must perform after translation:
  - Verify every original timestamp line (regex '-->') exists unchanged and in the same order.
  - Verify the number of subtitle blocks and blank-line separators hasn't changed.
  - If the model prepends an explanatory prefix (e.g., 'Voici le fichier...'), strip it before saving.
  - For very long files, prefer chunked/batched translations or use the Google fallback to avoid token/context issues.

- Short strict instruction to include on every translation request:

  "Return ONLY the translated file. Keep timestamps exactly. Do not output any commentary, labels, or analysis."


If any section is unclear or you want more examples (e.g., prompt text patterns or a checklist for reviewers), tell me which part to expand and I'll iterate. 🔧