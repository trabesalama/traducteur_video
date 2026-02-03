import importlib.util
import asyncio
spec = importlib.util.spec_from_file_location("dub4", r"c:\Users\ratah\OneDrive\Documents\traducteur\dub4.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

async def s():
    ssml = '<speak>Hello testing stream</speak>'
    com = mod.edge_tts.Communicate(ssml, 'fr-FR-HenriNeural')
    i = 0
    async for ch in com.stream():
        i += 1
        print('chunk', i, 'type:', ch.get('type'))
        if ch.get('type') == 'audio':
            data = ch.get('data')
            if data:
                import base64
                raw = base64.b64decode(data)
                print('audio len b64:', len(data), 'raw len:', len(raw), 'first bytes:', raw[:16])
        if i > 8:
            break

if __name__ == '__main__':
    asyncio.run(s())
