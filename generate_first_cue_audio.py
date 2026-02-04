import asyncio
import os
from pydub import AudioSegment
import edge_tts

SRC_SUB = "subtitle_first_fr.srt"
OUT_AUDIO = "first_cue_6s.mp3"
VOICE = "fr-FR-DeniseNeural"


def extract_first_cue(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if "-->" in line:
            # parse times
            times = line
            # collect text lines after
            i += 1
            text_lines = []
            while i < n and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines).strip()
            # parse end time for duration if needed
            try:
                left, right = times.split("-->")
                start = left.strip()
                end = right.strip().split()[0]
            except Exception:
                start = None
                end = None
            return start, end, text
        i += 1
    return None, None, None


async def synthesize_to_path(text: str, out_path: str):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(out_path)


def time_stretch_to_duration(in_path: str, out_path: str, desired_ms: int):
    seg = AudioSegment.from_file(in_path)
    current_ms = len(seg)
    if current_ms == 0:
        stretched = AudioSegment.silent(duration=desired_ms)
    else:
        speed = current_ms / float(desired_ms)
        new_frame_rate = int(seg.frame_rate * speed)
        stretched = seg._spawn(seg.raw_data, overrides={"frame_rate": new_frame_rate})
        stretched = stretched.set_frame_rate(seg.frame_rate)
        if len(stretched) > desired_ms:
            stretched = stretched[:desired_ms]
        else:
            stretched = stretched + AudioSegment.silent(duration=desired_ms - len(stretched))
    stretched.export(out_path, format="mp3")


def main():
    if not os.path.exists(SRC_SUB):
        print(f"Fichier de sous-titres introuvable: {SRC_SUB}")
        return

    start, end, text = extract_first_cue(SRC_SUB)
    if not text:
        print("Aucun texte trouvé dans le premier bloc.")
        return

    print(f"Premier bloc détecté: '{text[:60]}...' -> synthèse...")

    temp_path = "first_cue_raw.mp3"
    asyncio.run(synthesize_to_path(text, temp_path))

    # Desired duration: if end-start present, compute; else default 6000ms
    desired_ms = 6000
    if start and end:
        # simple parse mm:ss,mmm or hh:mm:ss,mmm
        def parse_ts(ts: str):
            ts = ts.replace(',', '.')
            parts = ts.split(':')
            parts = [float(p) for p in parts]
            if len(parts) == 3:
                return parts[0]*3600 + parts[1]*60 + parts[2]
            if len(parts) == 2:
                return parts[0]*60 + parts[1]
            return float(parts[0])

        try:
            s = parse_ts(start)
            e = parse_ts(end)
            desired_ms = int(max(0.1, e - s) * 1000)
        except Exception:
            desired_ms = 6000

    print(f"Ajustement audio à {desired_ms} ms...")
    time_stretch_to_duration(temp_path, OUT_AUDIO, desired_ms)

    # cleanup temp
    try:
        os.remove(temp_path)
    except Exception:
        pass

    print(f"Audio généré: {OUT_AUDIO} ({desired_ms} ms)")


if __name__ == '__main__':
    main()
