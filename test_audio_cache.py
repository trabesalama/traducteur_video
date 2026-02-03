import importlib.util, asyncio
spec = importlib.util.spec_from_file_location("dub4", r"c:\Users\ratah\OneDrive\Documents\traducteur\dub4.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

async def run():
    sem = asyncio.BoundedSemaphore(1)
    text = "Bonjour, ceci est un test pour le cache audio"
    print('First generate (should miss)')
    r1 = await mod.generate_tts_synced(text, 'fr-FR-HenriNeural', 2.0, None, sem, timeout_per_attempt=30, max_attempts=2, return_bytes=True)
    print('Audio size:', len(r1.get('audio_bytes') or b''))
    print('AUDIO_CACHE_HITS', mod.AUDIO_CACHE_HITS, 'MISSES', mod.AUDIO_CACHE_MISSES)

    print('Second generate (should hit)')
    r2 = await mod.generate_tts_synced(text, 'fr-FR-HenriNeural', 2.0, None, sem, timeout_per_attempt=30, max_attempts=2, return_bytes=True)
    print('Audio size:', len(r2.get('audio_bytes') or b''))
    print('AUDIO_CACHE_HITS', mod.AUDIO_CACHE_HITS, 'MISSES', mod.AUDIO_CACHE_MISSES)

if __name__ == '__main__':
    asyncio.run(run())