import importlib.util, asyncio, io
spec = importlib.util.spec_from_file_location("dub4", r"c:\Users\ratah\OneDrive\Documents\traducteur\dub4.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

async def test_bytes():
    print('Requesting raw bytes from tts_save_ssml...')
    b = await mod.tts_save_ssml('Bonjour test bytes', 'fr-FR-HenriNeural', 100, None, return_bytes=True)
    print('Bytes len:', len(b) if b else None)
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(io.BytesIO(b), format='mp3')
        print('pydub duration (s):', len(audio)/1000.0)
    except Exception as e:
        print('Error parsing bytes:', e)

if __name__ == '__main__':
    asyncio.run(test_bytes())
