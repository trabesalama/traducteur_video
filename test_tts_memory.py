import importlib.util
import asyncio
spec = importlib.util.spec_from_file_location("dub4", r"c:\Users\ratah\OneDrive\Documents\traducteur\dub4.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

async def test():
    sem = asyncio.BoundedSemaphore(1)
    print("Testing in-memory TTS generation (small phrase)...")
    res = await mod.generate_tts_synced("Bonjour, ceci est un test pour vérifier la génération audio en mémoire.", "fr-FR-HenriNeural", 2.0, None, sem, timeout_per_attempt=30, max_attempts=2, return_bytes=True)
    print("Result keys:", list(res.keys()))
    print("Duration:", res.get('duration'))
    if res.get('audio_bytes'):
        print("Audio bytes size:", len(res.get('audio_bytes')))
    else:
        print("No audio bytes returned")

if __name__ == '__main__':
    asyncio.run(test())
